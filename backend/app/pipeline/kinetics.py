"""Kinetic chain timing: pelvis → trunk → shoulder → elbow → wrist."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import savgol_filter


@dataclass
class KineticChainTiming:
    """Peak angular/linear velocity frames for each segment in the kinetic chain."""

    pelvis_peak_frame: int
    trunk_peak_frame: int
    shoulder_peak_frame: int
    elbow_peak_frame: int
    wrist_peak_frame: int

    @property
    def sequence_valid(self) -> bool:
        """True if pelvis peaks before trunk peaks before shoulder etc."""
        return (
            self.pelvis_peak_frame
            <= self.trunk_peak_frame
            <= self.shoulder_peak_frame
            <= self.elbow_peak_frame
            <= self.wrist_peak_frame
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "pelvis_peak_frame": self.pelvis_peak_frame,
            "trunk_peak_frame": self.trunk_peak_frame,
            "shoulder_peak_frame": self.shoulder_peak_frame,
            "elbow_peak_frame": self.elbow_peak_frame,
            "wrist_peak_frame": self.wrist_peak_frame,
        }


def _segment_angular_velocity(
    distal: NDArray[np.float64],
    proximal: NDArray[np.float64],
    fps: float = 30.0,
    window: int = 7,
) -> NDArray[np.float64]:
    """Approximate angular velocity of a segment as rate of change of its direction.

    Args:
        distal: (T, 3) distal joint positions.
        proximal: (T, 3) proximal joint positions.
        fps: frames per second.
        window: Savitzky-Golay window length.

    Returns:
        (T,) angular velocity magnitude in deg/s.
    """
    T = distal.shape[0]
    wl = window if T >= window else (T if T % 2 == 1 else max(3, T - 1))
    polyorder = min(3, wl - 1)

    seg = distal - proximal  # (T, 3)
    norms = np.linalg.norm(seg, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    unit = seg / norms  # (T, 3) unit vectors

    if T >= wl:
        unit = savgol_filter(unit, window_length=wl, polyorder=polyorder, axis=0)

    d_unit = np.gradient(unit, 1.0 / fps, axis=0)
    omega = np.degrees(np.linalg.norm(d_unit, axis=1))
    return omega


def _speed(
    positions: NDArray[np.float64],
    fps: float = 30.0,
    window: int = 7,
) -> NDArray[np.float64]:
    """Smoothed linear speed of a joint."""
    T = positions.shape[0]
    wl = window if T >= window else (T if T % 2 == 1 else max(3, T - 1))
    polyorder = min(3, wl - 1)
    if T >= wl:
        smoothed = savgol_filter(positions, window_length=wl, polyorder=polyorder, axis=0)
    else:
        smoothed = positions
    vel = np.gradient(smoothed, 1.0 / fps, axis=0)
    return np.linalg.norm(vel, axis=1)


def compute_kinetic_chain(
    joints_mhr70: NDArray[np.float64],
    left_hip_idx: int,
    right_hip_idx: int,
    left_shoulder_idx: int,
    right_shoulder_idx: int,
    throw_shoulder_idx: int,
    throw_elbow_idx: int,
    throw_wrist_idx: int,
    foot_plant_frame: int,
    fps: float = 30.0,
    search_end_frame: int | None = None,
) -> KineticChainTiming:
    """Compute kinetic chain timing by finding peak velocities for each link.

    Segments:
      Pelvis:    angular velocity of (right_hip - left_hip) vector
      Trunk:     angular velocity of (right_shoulder - left_shoulder) vector
      Shoulder:  angular velocity of (elbow - shoulder) vector
      Elbow:     angular velocity of (wrist - elbow) vector
      Wrist:     linear speed of wrist joint

    All peaks are searched from foot_plant_frame onward.

    Args:
        joints_mhr70: (T, 70, 3).
        *_idx: joint indices in MHR70.
        foot_plant_frame: start of search window.
        fps: frames per second.
        search_end_frame: end of search window (defaults to T).

    Returns:
        KineticChainTiming with per-segment peak frames.
    """
    T = joints_mhr70.shape[0]
    start = max(0, foot_plant_frame)
    end = search_end_frame if search_end_frame is not None else T

    def peak_in_window(signal: NDArray[np.float64]) -> int:
        chunk = signal[start:end]
        if len(chunk) == 0:
            return start
        return start + int(np.argmax(chunk))

    pelvis_omega = _segment_angular_velocity(
        joints_mhr70[:, right_hip_idx],
        joints_mhr70[:, left_hip_idx],
        fps=fps,
    )
    trunk_omega = _segment_angular_velocity(
        joints_mhr70[:, right_shoulder_idx],
        joints_mhr70[:, left_shoulder_idx],
        fps=fps,
    )
    shoulder_omega = _segment_angular_velocity(
        joints_mhr70[:, throw_elbow_idx],
        joints_mhr70[:, throw_shoulder_idx],
        fps=fps,
    )
    elbow_omega = _segment_angular_velocity(
        joints_mhr70[:, throw_wrist_idx],
        joints_mhr70[:, throw_elbow_idx],
        fps=fps,
    )
    wrist_speed = _speed(joints_mhr70[:, throw_wrist_idx], fps=fps)

    return KineticChainTiming(
        pelvis_peak_frame=peak_in_window(pelvis_omega),
        trunk_peak_frame=peak_in_window(trunk_omega),
        shoulder_peak_frame=peak_in_window(shoulder_omega),
        elbow_peak_frame=peak_in_window(elbow_omega),
        wrist_peak_frame=peak_in_window(wrist_speed),
    )
