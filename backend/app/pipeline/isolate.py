"""Pitcher isolation via center-crop heuristic."""

from __future__ import annotations

import numpy as np


def isolate_pitcher(frame: np.ndarray) -> np.ndarray:
    """Crop to the center 60% of frame width, full height.

    Pitchers appear in the center third of most broadcast and bullpen footage.
    Full person detection (Faster R-CNN) is handled in inference.py; this is
    a fast pre-crop to reduce the detection search area.

    Args:
        frame: RGB uint8 array of shape (H, W, 3).

    Returns:
        Center-cropped frame (H, ~0.6*W, 3).
    """
    h, w = frame.shape[:2]
    crop_w = int(w * 0.60)
    x_start = (w - crop_w) // 2
    return frame[:, x_start : x_start + crop_w, :]
