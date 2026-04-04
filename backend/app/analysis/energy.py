"""Energy decomposition at the release point."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def decompose_energy(
    release_point_series: NDArray,
    fps: float = 30.0,
) -> dict[str, float]:
    """Decompose hand/wrist velocity at release into directional components.

    Args:
        release_point_series: (T, 3) wrist positions over time.
        fps: video frame rate.

    Returns:
        Dict with forward, lateral, vertical, total, wasted_motion_index.
    """
    arr = np.array(release_point_series, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] < 2:
        return {
            "forward": 0.0,
            "lateral": 0.0,
            "vertical": 0.0,
            "total": 0.0,
            "wasted_motion_index": 0.0,
        }

    # Use last min(5, T) frames for velocity estimate
    n = min(5, arr.shape[0])
    tail = arr[-n:]
    if n < 2:
        vel = np.zeros(3)
    else:
        vel = np.gradient(tail, 1.0 / fps, axis=0).mean(axis=0)

    forward = abs(float(vel[2]))   # z = toward home plate
    lateral = abs(float(vel[0]))   # x = arm-side / glove-side
    vertical = abs(float(vel[1]))  # y = up/down

    total = float(np.linalg.norm(vel))
    if total < 1e-9:
        wasted = 0.0
    else:
        wasted = 1.0 - (forward / total)

    return {
        "forward": forward,
        "lateral": lateral,
        "vertical": vertical,
        "total": total,
        "wasted_motion_index": wasted,
    }
