"""Pitch-to-pitch temporal alignment.

Aligns multiple pitch sequences to a common reference frame so that
biomechanical features can be compared across pitches.

Reference points (in order of preference): foot plant, MER, release.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from backend.app.pipeline.phases import PitchPhases


def align_to_event(
    joints: NDArray[np.float64],
    phases: PitchPhases,
    anchor: str = "mer",
    pre_frames: int = 20,
    post_frames: int = 20,
) -> NDArray[np.float64]:
    """Clip and return a window of frames centered on a phase event.

    Args:
        joints: (T, J, 3) joint array.
        phases: PitchPhases for this sequence.
        anchor: "foot_plant" | "mer" | "release"
        pre_frames: frames before anchor to include.
        post_frames: frames after anchor to include.

    Returns:
        (pre_frames + 1 + post_frames, J, 3) array, zero-padded if needed.
    """
    T, J, _ = joints.shape
    anchor_map = {
        "foot_plant": phases.foot_plant,
        "mer": phases.mer,
        "release": phases.release,
    }
    if anchor not in anchor_map:
        raise ValueError(f"Unknown anchor: {anchor!r}")

    center = anchor_map[anchor]
    window_len = pre_frames + 1 + post_frames
    out = np.zeros((window_len, J, 3), dtype=np.float64)

    for i in range(window_len):
        src = center - pre_frames + i
        if 0 <= src < T:
            out[i] = joints[src]

    return out


def normalize_time_axis(
    joints: NDArray[np.float64],
    phases: PitchPhases,
    n_samples: int = 101,
) -> NDArray[np.float64]:
    """Resample the pitch sequence to a fixed number of frames using phase boundaries.

    The sequence is divided into three segments:
        [0, foot_plant] → [0, 33] (wind-up)
        [foot_plant, mer] → [33, 66] (arm cocking)
        [mer, release] → [66, 100] (acceleration)

    Each segment is linearly interpolated to its target length.

    Args:
        joints: (T, J, 3) joint array.
        phases: PitchPhases for this sequence.
        n_samples: output frame count (must be divisible by 3 cleanly enough).

    Returns:
        (n_samples, J, 3) resampled array.
    """
    T, J, _ = joints.shape
    seg_len = n_samples // 3
    remainder = n_samples - 3 * seg_len

    boundaries = [
        (0, phases.foot_plant, seg_len),
        (phases.foot_plant, phases.mer, seg_len),
        (phases.mer, phases.release, seg_len + remainder),
    ]

    segments = []
    for start, end, target in boundaries:
        start = max(0, min(start, T - 1))
        end = max(start + 1, min(end + 1, T))
        chunk = joints[start:end]  # (chunk_len, J, 3)
        chunk_len = chunk.shape[0]
        if chunk_len == target:
            segments.append(chunk)
        else:
            # Resample each joint coordinate independently via linspace
            src_t = np.linspace(0, 1, chunk_len)
            dst_t = np.linspace(0, 1, target)
            resampled = np.zeros((target, J, 3))
            for j in range(J):
                for d in range(3):
                    resampled[:, j, d] = np.interp(dst_t, src_t, chunk[:, j, d])
            segments.append(resampled)

    return np.concatenate(segments, axis=0)
