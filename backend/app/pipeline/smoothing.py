"""Temporal smoothing for joint trajectories."""

from __future__ import annotations

import numpy as np


def smooth_joints(joints: np.ndarray, method: str = "kalman") -> np.ndarray:
    """Smooth a joint trajectory over time.

    Args:
        joints: Float array of shape (T, J, 3).
        method: "kalman" | "butterworth" | "none".

    Returns:
        Smoothed array of shape (T, J, 3).
    """
    if method == "none":
        return joints

    if method == "kalman":
        return _smooth_kalman(joints)

    if method == "butterworth":
        return _smooth_butterworth(joints)

    raise ValueError(f"Unknown smoothing method: {method!r}. Use 'kalman', 'butterworth', or 'none'.")


def _smooth_kalman(joints: np.ndarray) -> np.ndarray:
    """Per-joint Kalman smoothing (constant-velocity model)."""
    from filterpy.kalman import KalmanFilter

    T, J, D = joints.shape
    out = np.zeros_like(joints)

    for j in range(J):
        for d in range(D):
            kf = KalmanFilter(dim_x=2, dim_z=1)
            kf.F = np.array([[1.0, 1.0], [0.0, 1.0]])
            kf.H = np.array([[1.0, 0.0]])
            kf.R = np.array([[1.0]])
            kf.Q = np.eye(2) * 0.01
            kf.x = np.array([[joints[0, j, d]], [0.0]])
            kf.P = np.eye(2)

            for t in range(T):
                kf.predict()
                kf.update(np.array([[joints[t, j, d]]]))
                out[t, j, d] = float(kf.x[0, 0])

    return out


def _smooth_butterworth(joints: np.ndarray) -> np.ndarray:
    """Per-coordinate Butterworth low-pass filter."""
    from scipy.signal import butter, filtfilt

    T, J, D = joints.shape
    if T < 9:
        # filtfilt requires padlen < signal length; fall back for very short clips
        return joints.copy()

    b, a = butter(N=4, Wn=0.1, btype="low")
    out = np.zeros_like(joints)

    for j in range(J):
        for d in range(D):
            out[:, j, d] = filtfilt(b, a, joints[:, j, d])

    return out
