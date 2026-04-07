"""SAM 3.1 pitcher detector wrapper around mlx-vlm.

Used during clip trimming (Stage 1 of the data pipeline) to identify the
first frame in a downloaded broadcast clip where the pitcher is in the
set position on the mound. Once that frame is found, the clip is trimmed
to a fixed window (set_frame, set_frame + 5 seconds) covering the entire
delivery from set position through landing of the stride foot.

This module is intentionally kept separate from the per-frame SAM 3D Body
inference loop. SAM 3.1 is too slow (~1000ms per frame at 1008px on M3 Max)
to run inside the per-frame loop, but only ~30 sampled frames need to be
checked per clip during trimming, so the total Stage 1 cost is ~30 seconds
per clip.

Public entry point:
    detector = get_pitcher_detector()
    bbox = detector.detect_pitcher(frame_rgb)
    # bbox is (x1, y1, x2, y2) or None if no pitcher found

Loads the model lazily on first use and caches it as a module-level singleton.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

_DEFAULT_MODEL_ID = "mlx-community/sam3.1-bf16"
_DEFAULT_PROMPT = "a baseball pitcher"
_DEFAULT_SCORE_THRESHOLD = 0.30

# Tiebreaker constraints for picking the pitcher among multiple uniformed people.
# A pitcher on the elevated mound, viewed from the standard center-field broadcast cam, is:
#   - Standing upright (aspect ratio: tall, height/width >= 1.5)
#   - In the lower half of the frame, BELOW the horizon line (>= 55% down)
#   - LARGE in the frame (>= 35% of frame height) because he's the closest person
#   - Centered (not at the very left/right edge where dugouts and fans live)
#
# These thresholds were calibrated against a 1280x720 broadcast frame from
# Darvish 2017 WS G7. The previous looser thresholds (>= 40% Y, >= 15% height)
# accidentally accepted the batter, catcher, umpire, AND fielders during
# post-contact field-action shots. The tightened values cleanly select only
# the pitcher across all sampled frames in the clip we tested. See
# scripts/test_delivery_window.py for the calibration data.
_MIN_ASPECT_RATIO = 1.5
_MIN_BBOX_HEIGHT_FRACTION = 0.35      # excludes catcher, umpire, batter, far fielders
_MAX_BBOX_HEIGHT_FRACTION = 0.85      # excludes a closeup of a glove or face
_LOWER_HALF_THRESHOLD = 0.55          # excludes batter (~0.43) and umpire (~0.43)
_CENTER_HALF_THRESHOLD = 0.20         # excludes dugout / fans at the edges


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class PitcherDetection:
    """Result of a single pitcher-detection call."""

    bbox: tuple[int, int, int, int]   # (x1, y1, x2, y2) in pixel coords
    score: float                       # SAM 3.1 confidence
    aspect_ratio: float                # height / width
    height_fraction: float             # bbox height / frame height
    center_x_fraction: float           # bbox center X / frame width
    center_y_fraction: float           # bbox center Y / frame height

    def to_dict(self) -> dict:
        return {
            "bbox": list(self.bbox),
            "score": self.score,
            "aspect_ratio": self.aspect_ratio,
            "height_fraction": self.height_fraction,
            "center_x_fraction": self.center_x_fraction,
            "center_y_fraction": self.center_y_fraction,
        }


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------

class Sam31PitcherDetector:
    """Wrapper around the mlx-vlm SAM 3.1 predictor with pitcher-specific tiebreaking."""

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL_ID,
        prompt: str = _DEFAULT_PROMPT,
        score_threshold: float = _DEFAULT_SCORE_THRESHOLD,
    ):
        self.model_id = model_id
        self.prompt = prompt
        self.score_threshold = score_threshold
        self._predictor = None  # lazy-loaded

    def _ensure_loaded(self) -> None:
        """Load SAM 3.1 into MLX memory on first use."""
        if self._predictor is not None:
            return
        logger.info("loading SAM 3.1 from %s ...", self.model_id)
        from mlx_vlm.models.sam3.generate import Sam3Predictor
        from mlx_vlm.models.sam3_1.processing_sam3_1 import Sam31Processor
        from mlx_vlm.utils import load_model, get_model_path

        model_path = get_model_path(self.model_id)
        model = load_model(model_path)
        processor = Sam31Processor.from_pretrained(str(model_path))
        self._predictor = Sam3Predictor(
            model, processor, score_threshold=self.score_threshold
        )
        logger.info("SAM 3.1 loaded.")

    def detect_pitcher(
        self,
        frame_rgb: NDArray,
        prompt: Optional[str] = None,
    ) -> Optional[PitcherDetection]:
        """Run SAM 3.1 on a single RGB frame and return the pitcher's bbox.

        Returns None if no detection passes the geometric constraints (no
        upright standing person in the lower half of the frame).
        """
        self._ensure_loaded()
        from PIL import Image as PILImage

        if frame_rgb.dtype != np.uint8:
            frame_rgb = frame_rgb.astype(np.uint8)
        image = PILImage.fromarray(frame_rgb)
        result = self._predictor.predict(image, text_prompt=prompt or self.prompt)

        n = len(result.scores) if hasattr(result, "scores") and result.scores is not None else 0
        if n == 0:
            return None

        h, w = frame_rgb.shape[:2]

        # Build candidate list with all the geometric features
        candidates: list[PitcherDetection] = []
        for i in range(n):
            x1, y1, x2, y2 = (int(v) for v in result.boxes[i])
            score = float(result.scores[i])
            bw = x2 - x1
            bh = y2 - y1
            if bw <= 0 or bh <= 0:
                continue
            aspect = bh / bw
            height_frac = bh / max(h, 1)
            cx_frac = (x1 + x2) / 2.0 / max(w, 1)
            cy_frac = (y1 + y2) / 2.0 / max(h, 1)

            # Geometric filters: must look like a standing pitcher
            if aspect < _MIN_ASPECT_RATIO:
                continue
            if height_frac < _MIN_BBOX_HEIGHT_FRACTION:
                continue
            if height_frac > _MAX_BBOX_HEIGHT_FRACTION:
                continue
            if cy_frac < _LOWER_HALF_THRESHOLD:
                continue
            if cx_frac < _CENTER_HALF_THRESHOLD or cx_frac > (1.0 - _CENTER_HALF_THRESHOLD):
                continue

            candidates.append(
                PitcherDetection(
                    bbox=(x1, y1, x2, y2),
                    score=score,
                    aspect_ratio=aspect,
                    height_fraction=height_frac,
                    center_x_fraction=cx_frac,
                    center_y_fraction=cy_frac,
                )
            )

        if not candidates:
            return None

        # Tiebreaker: largest area among the geometric survivors
        def area(d: PitcherDetection) -> float:
            x1, y1, x2, y2 = d.bbox
            return float((x2 - x1) * (y2 - y1))

        candidates.sort(key=area, reverse=True)
        return candidates[0]


# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------

_cached_detector: Optional[Sam31PitcherDetector] = None


def get_pitcher_detector(
    model_id: str = _DEFAULT_MODEL_ID,
    prompt: str = _DEFAULT_PROMPT,
) -> Sam31PitcherDetector:
    """Return the module-level cached pitcher detector, instantiating on first call."""
    global _cached_detector
    if _cached_detector is None:
        _cached_detector = Sam31PitcherDetector(model_id=model_id, prompt=prompt)
    return _cached_detector
