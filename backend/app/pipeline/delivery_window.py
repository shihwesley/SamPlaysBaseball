"""Dynamic delivery-window detection for downloaded broadcast pitch clips.

Stage 1 of the data pipeline. Called from `fetch_savant_clips.py` immediately
after each clip downloads. Output: a trimmed .mp4 containing only the actual
pitcher delivery (set position through landing of the stride foot).

The problem this solves
-----------------------
Per-pitch broadcast clips are roughly 10 seconds long. The actual pitcher
delivery occupies a sub-window. The lead-in may show graphics, the broadcast
booth, the previous play wrapping up, or a batter close-up. The trailing
seconds often show field action after the ball is hit (broadcasts cut to
follow runners and fielders) or replay overlays. The exact framing varies
across:

  - 2017 postseason broadcasts (HD, slower cuts, earlier graphic packages)
  - Modern (2025+) broadcasts (faster cuts, PIPs, K-zone overlays)
  - Amateur and minor-league footage (varying camera quality and framing)
  - Practice / bullpen footage (no broadcast cuts at all)

A hardcoded "first N frames" trim would break the moment we hit a clip with
a different broadcast pattern. We need a dynamic detector that finds the
actual delivery regardless of pre-pitch lead-in length or post-contact
behavior.

Algorithm
---------
The detector uses two complementary signals — each one for what it's best at:

  Phase 1 — SAM 3.1 (semantic): find the set frame
    Scan forward at a coarse stride, running SAM 3.1 with the prompt
    "a baseball pitcher" + the geometric pitcher tiebreaker (lower-half of
    frame, standing aspect ratio, large height fraction). Stop as soon as we
    have two consecutive sampled frames with a positive pitcher detection.
    The frame_index of the FIRST of those two is the set frame.
    Cost: ~2-5 SAM 3.1 calls per clip = ~2-5 seconds.

  Phase 2 — cv2 histogram diff (structural): find any broadcast cut
    Scan frames [set_frame, set_frame + biomech_max] with color histogram
    correlation. The first correlation drop below a threshold = scene cut.
    This catches the case where the broadcast cut to field action mid-delivery
    (hit-into-play pitches) without needing more SAM calls.
    Cost: ~0.2 seconds for the post-set range.

  Phase 3 — Combine
    biomech_end = set_frame + round(5.0 * fps)   (FPS-aware)
    end_frame   = min(biomech_end, scene_cut, total_frames)
    start_frame = max(0, set_frame - round(0.5 * fps))

Total cost per clip: ~3-5 seconds (vs ~60s for the previous uniform-sampling
approach). For a 47-pitch game that's ~3 minutes of trim work instead of ~47.

Public entry points:
    find_delivery_window(video_path) -> DeliveryWindow
    write_trimmed_clip(source, dest, start, end)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------

# Phase 1 — SAM 3.1 set-frame scan
# Time between samples in seconds. The detector should sample frequently
# enough to catch the set position (which lasts 1-2 seconds) but not so
# frequently that we waste SAM calls. 0.2s = 5 samples per second.
_SAMPLE_INTERVAL_SECONDS = 0.2

# Minimum consecutive sampled frames with a positive pitcher detection
# required before accepting the first as the set frame.
#
# Set to 1 because the geometric filter in sam3d_mlx.sam31_detector is now
# tight enough that false positives are very rare (would require a non-pitcher
# person at >35% frame height in the lower half of the center-frame, which
# excludes catcher, umpire, batter, all fielders, and crowd). Empirically:
# raising to 2 caused legitimate set frames to be missed when SAM 3.1 had
# borderline scores on 1-2 consecutive samples in the set position. The
# downstream scene-cut detection acts as a second line of defense.
_MIN_CONSECUTIVE_DETECTIONS = 1

# Maximum number of seconds of source video to scan before giving up the
# search for a set frame. Most broadcast clips have the pitcher in frame
# within 3-4 seconds. We scan up to this limit so we don't waste time on
# clips that never show the pitcher.
_MAX_SCAN_SECONDS = 8.0

# Phase 2 — cv2 scene-cut detection in the post-set range
#
# Calibrated against the Darvish 2017 WS G7 in-play CH clip:
#   - Pitcher-cam frames have inter-frame correlations of 0.99-1.00 (very stable)
#   - The actual broadcast cut to wide field shot dropped correlation to 0.649
#   - Threshold 0.85 sits comfortably between (catches the cut, doesn't fire on noise)
# This is much more lenient than scene-detection literature defaults (~0.55)
# because broadcast cuts in baseball footage often share the same
# field-green dominant color, making the histogram distance smaller than
# in scripted-content scene cuts.
_HIST_BINS = (8, 8, 8)
_HIST_DOWNSAMPLE_TO = (160, 90)
_HIST_CORR_CUT_THRESHOLD = 0.85

# Phase 3 — Trim window framing (in seconds, FPS-aware)
# Set position → leg lift → stride → release → follow-through → landing
# is biomechanically ~3-4 seconds. 5.0 gives a 1-2s safety margin.
_PRE_SET_BUFFER_SECONDS = 0.5
_POST_SET_DELIVERY_SECONDS = 5.0
_POST_SCENE_CUT_BUFFER_SECONDS = 0.0  # don't extend past a cut


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SampledFrame:
    """Per-sample debug record from the SAM scan."""

    frame_index: int
    pitcher_detected: bool
    bbox: Optional[tuple[int, int, int, int]] = None
    score: Optional[float] = None
    note: str = ""


@dataclass
class DeliveryWindow:
    """Result of delivery-window detection.

    Attributes:
        start_frame: Inclusive starting frame of the trimmed window.
        end_frame:   Exclusive ending frame of the trimmed window.
        total_frames: Total frame count of the source video.
        fps: Source video frame rate.
        set_frame: The detected first-set-position frame index.
        scene_cut_frame: Frame index of detected broadcast cut after set,
                         or None if no cut was found within the biomech window.
        confidence: 0..1 score. 1.0 = clean detection of the set frame and
                    a clean post-set window. 0.0 = no set frame found.
        sampled: Per-sample debug records from the SAM scan.
    """

    start_frame: int
    end_frame: int
    total_frames: int
    fps: float
    set_frame: Optional[int]
    scene_cut_frame: Optional[int]
    confidence: float
    sampled: list[SampledFrame] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return max(0, self.end_frame - self.start_frame)

    @property
    def duration_seconds(self) -> float:
        return self.n_frames / self.fps if self.fps > 0 else 0.0

    @property
    def trimmed(self) -> bool:
        """True if any frames were trimmed off (start>0 or end<total)."""
        return self.start_frame > 0 or self.end_frame < self.total_frames

    def to_dict(self) -> dict:
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "total_frames": self.total_frames,
            "fps": self.fps,
            "set_frame": self.set_frame,
            "scene_cut_frame": self.scene_cut_frame,
            "n_frames_kept": self.n_frames,
            "duration_kept_s": self.duration_seconds,
            "confidence": self.confidence,
            "n_sam_calls": len(self.sampled),
        }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def find_delivery_window(
    video_path: str | Path,
    detector=None,
    sample_interval_s: float = _SAMPLE_INTERVAL_SECONDS,
    max_scan_s: float = _MAX_SCAN_SECONDS,
    pre_set_buffer_s: float = _PRE_SET_BUFFER_SECONDS,
    post_set_delivery_s: float = _POST_SET_DELIVERY_SECONDS,
    min_consecutive: int = _MIN_CONSECUTIVE_DETECTIONS,
) -> DeliveryWindow:
    """Find the pitcher delivery window in a downloaded broadcast clip.

    Args:
        video_path: Path to the source clip.
        detector: Optional Sam31PitcherDetector instance. Defaults to the
                  module-cached singleton from sam3d_mlx.sam31_detector.
        sample_interval_s: Time between samples in the SAM scan. 0.2s by
                  default = 5 samples per second of source video, FPS-aware.
        max_scan_s: Maximum source seconds to scan when looking for the
                  set frame. Most clips have the pitcher within 3-4s.
        pre_set_buffer_s: Seconds of pre-set footage to keep in the trim.
        post_set_delivery_s: Seconds of post-set footage to keep. The
                  biomechanical maximum for a pitch delivery is ~4-5 seconds
                  from set to landing of the stride foot.
        min_consecutive: Minimum consecutive sampled frames with a positive
                  pitcher detection required to accept the set frame.

    Returns:
        DeliveryWindow with the trim bounds and debug data. If no set frame
        is found, returns a window covering the entire clip with confidence
        0.0 so the caller can fall back to no trim.
    """
    video_path = Path(video_path)
    if detector is None:
        from sam3d_mlx.sam31_detector import get_pitcher_detector
        detector = get_pitcher_detector()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    # ---------- Phase 1: early-exit SAM scan for the set frame ----------
    sampled, set_frame = _early_exit_set_frame_scan(
        video_path=video_path,
        detector=detector,
        total_frames=total_frames,
        fps=fps,
        sample_interval_s=sample_interval_s,
        max_scan_s=max_scan_s,
        min_consecutive=min_consecutive,
    )

    if set_frame is None:
        logger.warning(
            "no pitcher set-frame detected in %s after %d SAM calls — returning full clip",
            video_path.name,
            len(sampled),
        )
        return DeliveryWindow(
            start_frame=0,
            end_frame=total_frames,
            total_frames=total_frames,
            fps=fps,
            set_frame=None,
            scene_cut_frame=None,
            confidence=0.0,
            sampled=sampled,
        )

    # ---------- Phase 3 (compute candidate window) ----------
    pre_buffer = int(round(pre_set_buffer_s * fps))
    delivery_window = int(round(post_set_delivery_s * fps))
    biomech_end = set_frame + delivery_window
    candidate_end = min(biomech_end, total_frames)
    start_frame = max(0, set_frame - pre_buffer)

    # ---------- Phase 2: cv2 scene-cut detection in [set_frame, candidate_end) ----------
    scene_cut = _find_scene_cut(video_path, set_frame, candidate_end)

    if scene_cut is not None:
        post_cut_buffer = int(round(_POST_SCENE_CUT_BUFFER_SECONDS * fps))
        end_frame = min(candidate_end, scene_cut + post_cut_buffer)
    else:
        end_frame = candidate_end

    # Confidence: based on set-frame confirmation strength + presence of cut
    n_pos = sum(1 for s in sampled if s.pitcher_detected)
    confidence = float(np.clip(n_pos / max(len(sampled), 1), 0.0, 1.0))

    return DeliveryWindow(
        start_frame=start_frame,
        end_frame=end_frame,
        total_frames=total_frames,
        fps=fps,
        set_frame=set_frame,
        scene_cut_frame=scene_cut,
        confidence=confidence,
        sampled=sampled,
    )


# ---------------------------------------------------------------------------
# Phase 1 — Early-exit SAM scan
# ---------------------------------------------------------------------------

def _early_exit_set_frame_scan(
    video_path: Path,
    detector,
    total_frames: int,
    fps: float,
    sample_interval_s: float,
    max_scan_s: float,
    min_consecutive: int,
) -> tuple[list[SampledFrame], Optional[int]]:
    """Scan forward sparsely until we find min_consecutive consecutive
    samples with a positive pitcher detection. Stop and return as soon as
    confirmed. Returns (sampled_records, set_frame_index_or_None).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    stride = max(1, int(round(sample_interval_s * fps)))
    max_scan_frames = min(total_frames, int(round(max_scan_s * fps)))

    sampled: list[SampledFrame] = []
    consecutive_run: list[int] = []  # frame indices in current consecutive run

    for fi in range(0, max_scan_frames, stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame_bgr = cap.read()
        if not ok:
            sampled.append(SampledFrame(frame_index=fi, pitcher_detected=False, note="read_failed"))
            consecutive_run = []
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        try:
            detection = detector.detect_pitcher(frame_rgb)
        except Exception as exc:
            logger.warning("detector failed at frame %d: %s", fi, exc)
            sampled.append(
                SampledFrame(frame_index=fi, pitcher_detected=False, note=f"err:{type(exc).__name__}")
            )
            consecutive_run = []
            continue

        if detection is None:
            sampled.append(SampledFrame(frame_index=fi, pitcher_detected=False))
            consecutive_run = []
            continue

        sampled.append(
            SampledFrame(
                frame_index=fi,
                pitcher_detected=True,
                bbox=detection.bbox,
                score=detection.score,
            )
        )
        consecutive_run.append(fi)
        if len(consecutive_run) >= min_consecutive:
            cap.release()
            return sampled, consecutive_run[0]

    cap.release()
    return sampled, None


# ---------------------------------------------------------------------------
# Phase 2 — cv2 scene-cut detection in the post-set window
# ---------------------------------------------------------------------------

def _find_scene_cut(
    video_path: Path,
    start_frame: int,
    end_frame: int,
) -> Optional[int]:
    """Scan [start_frame, end_frame) for the first histogram-correlation drop
    indicating a broadcast cut. Returns the frame index of the cut, or None.
    """
    if end_frame - start_frame < 2:
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ok, prev = cap.read()
    if not ok:
        cap.release()
        return None

    prev_small = cv2.resize(prev, _HIST_DOWNSAMPLE_TO)
    prev_hist = cv2.calcHist(
        [prev_small], [0, 1, 2], None, _HIST_BINS, [0, 256, 0, 256, 0, 256]
    )
    cv2.normalize(prev_hist, prev_hist)

    cut_index: Optional[int] = None
    for fi in range(start_frame + 1, end_frame):
        ok, curr = cap.read()
        if not ok:
            break
        curr_small = cv2.resize(curr, _HIST_DOWNSAMPLE_TO)
        curr_hist = cv2.calcHist(
            [curr_small], [0, 1, 2], None, _HIST_BINS, [0, 256, 0, 256, 0, 256]
        )
        cv2.normalize(curr_hist, curr_hist)
        corr = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL)
        if corr < _HIST_CORR_CUT_THRESHOLD:
            cut_index = fi
            break
        prev_hist = curr_hist

    cap.release()
    return cut_index


# ---------------------------------------------------------------------------
# Helper: write a trimmed clip to disk
# ---------------------------------------------------------------------------

def write_trimmed_clip(
    source_video_path: str | Path,
    output_video_path: str | Path,
    start_frame: int,
    end_frame: int,
) -> int:
    """Write frames [start_frame, end_frame) of the source video to a new MP4.

    Uses cv2.VideoWriter with mp4v codec. Preserves source fps and resolution.
    Returns the number of frames written. Raises RuntimeError on failure.
    """
    source_video_path = Path(source_video_path)
    output_video_path = Path(output_video_path)

    cap = cv2.VideoCapture(str(source_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open source: {source_video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open writer: {output_video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    written = 0
    for _ in range(end_frame - start_frame):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        written += 1

    cap.release()
    writer.release()

    if written == 0:
        raise RuntimeError(
            f"no frames written from {source_video_path} "
            f"[{start_frame}, {end_frame}) → {output_video_path}"
        )
    logger.info(
        "trimmed %s [%d:%d) → %s (%d frames)",
        source_video_path.name,
        start_frame,
        end_frame,
        output_video_path.name,
        written,
    )
    return written
