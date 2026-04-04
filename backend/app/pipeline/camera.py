"""Camera parameter estimation and override."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CameraParams:
    focal_length: float
    cx: float
    cy: float
    cam_t: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 2.5]))


def estimate_camera(frame_width: int, frame_height: int) -> CameraParams:
    """Estimate camera intrinsics from frame dimensions.

    Uses max(width, height) as focal length — a standard heuristic for
    cameras without known calibration data.
    """
    return CameraParams(
        focal_length=float(max(frame_width, frame_height)),
        cx=frame_width / 2.0,
        cy=frame_height / 2.0,
        cam_t=np.array([0.0, 0.0, 2.5]),
    )


def override_camera(
    params: CameraParams,
    focal_length: float | None = None,
    cam_t: np.ndarray | None = None,
) -> CameraParams:
    """Return a new CameraParams with overridden values."""
    return CameraParams(
        focal_length=focal_length if focal_length is not None else params.focal_length,
        cx=params.cx,
        cy=params.cy,
        cam_t=cam_t if cam_t is not None else params.cam_t.copy(),
    )
