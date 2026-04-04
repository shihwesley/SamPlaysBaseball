"""Source-specific frame preprocessing."""

from __future__ import annotations

import cv2
import numpy as np

from backend.app.pipeline.video import SourceType


def preprocess_frame(frame: np.ndarray, source_type: SourceType) -> np.ndarray:
    """Apply source-specific preprocessing to a single RGB frame.

    Args:
        frame: RGB uint8 array of shape (H, W, 3).
        source_type: Video source type.

    Returns:
        Preprocessed frame, same shape as input.
    """
    if source_type == SourceType.broadcast:
        h, w = frame.shape[:2]
        top = int(h * 0.10)
        bottom = int(h * 0.90)
        cropped = frame[top:bottom, :, :]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    if source_type == SourceType.smartphone:
        return cv2.GaussianBlur(frame, (3, 3), sigmaX=0.5, sigmaY=0.5)

    if source_type == SourceType.milb:
        # CLAHE on L channel of LAB
        lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # bullpen: clean footage, no processing needed
    return frame
