"""Pitch delivery segmentation via frame differencing."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PitchSegment:
    frame_start: int
    frame_end: int
    confidence: float


def segment_pitch(frames: list[dict], fps: int = 30) -> list[PitchSegment]:
    """Detect pitch delivery segments from extracted frames.

    Uses frame differencing: computes mean absolute diff between consecutive
    grayscale frames, smooths with a rolling window, then finds motion peaks.

    Args:
        frames: List of dicts with "frame" key (H,W,3 RGB uint8).
        fps: Frames per second (used for padding constants).

    Returns:
        List of PitchSegment sorted by frame_start.
    """
    if len(frames) < 2:
        return []

    # Compute per-frame diff signal
    diffs: list[float] = [0.0]
    prev_gray = cv2.cvtColor(frames[0]["frame"], cv2.COLOR_RGB2GRAY).astype(np.float32)
    for fd in frames[1:]:
        gray = cv2.cvtColor(fd["frame"], cv2.COLOR_RGB2GRAY).astype(np.float32)
        diffs.append(float(np.mean(np.abs(gray - prev_gray))))
        prev_gray = gray

    diff_arr = np.array(diffs, dtype=np.float64)

    # Rolling mean smoothing (window=5)
    window = 5
    kernel = np.ones(window) / window
    smoothed = np.convolve(diff_arr, kernel, mode="same")

    # Find peaks above mean + 1.5*std
    threshold = smoothed.mean() + 1.5 * smoothed.std()
    peaks = [i for i, v in enumerate(smoothed) if v > threshold]

    if not peaks:
        return []

    # Group peaks within 15 frames into segments
    groups: list[list[int]] = [[peaks[0]]]
    for p in peaks[1:]:
        if p - groups[-1][-1] <= 15:
            groups[-1].append(p)
        else:
            groups.append([p])

    segments: list[PitchSegment] = []
    n = len(frames)
    for group in groups:
        start = max(0, group[0] - 10)
        end = min(n - 1, group[-1] + 20)
        # Confidence: normalized peak height above threshold
        peak_val = max(smoothed[i] for i in group)
        conf = min(1.0, float((peak_val - threshold) / (smoothed.max() - threshold + 1e-8)))
        segments.append(PitchSegment(frame_start=start, frame_end=end, confidence=conf))

    return sorted(segments, key=lambda s: s.frame_start)
