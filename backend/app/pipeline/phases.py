"""Pitch phase detection from 3D joint trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import savgol_filter


@dataclass
class PitchPhases:
    """Frame indices marking key delivery events."""

    # Frame where front foot vertical velocity decelerates to near zero (foot plant).
    foot_plant: int
    # Frame of peak shoulder external rotation.
    mer: int
    # Frame of ball release (wrist max forward extension velocity).
    release: int
    # Total frames in the sequence.
    total_frames: int

    @property
    def valid(self) -> bool:
        """True if phase order is sensible."""
        return self.foot_plant <= self.mer <= self.release


def _smooth_velocity(
    positions: NDArray[np.float64],
    fps: float = 30.0,
    window: int = 7,
    polyorder: int = 3,
) -> NDArray[np.float64]:
    """Compute smoothed velocity magnitude from position sequence.

    Args:
        positions: (T, 3) position array.
        fps: frames per second.
        window: Savitzky-Golay window length (must be odd).

    Returns:
        (T,) velocity magnitudes in units/s.
    """
    T = positions.shape[0]
    if T < window:
        window = T if T % 2 == 1 else max(3, T - 1)
        if window < 3:
            return np.zeros(T)
    smoothed = savgol_filter(positions, window_length=window, polyorder=polyorder, axis=0)
    vel = np.gradient(smoothed, 1.0 / fps, axis=0)
    return np.linalg.norm(vel, axis=1)


def detect_foot_plant(
    ankle_positions: NDArray[np.float64],
    fps: float = 30.0,
    window: int = 7,
    search_start_frac: float = 0.2,
    search_end_frac: float = 0.7,
) -> int:
    """Detect front-foot plant frame.

    Strategy: find the frame within the search window where vertical (y-axis)
    velocity of the stride ankle decelerates to its minimum (near zero),
    indicating contact.

    Args:
        ankle_positions: (T, 3) stride-ankle positions.
        fps: frames per second.
        search_start_frac: fraction of total frames to start searching.
        search_end_frac: fraction of total frames to end searching.

    Returns:
        Frame index of foot plant.
    """
    T = ankle_positions.shape[0]
    start = max(0, int(T * search_start_frac))
    end = min(T - 1, int(T * search_end_frac))

    # Vertical velocity of ankle
    y_pos = ankle_positions[:, 1]
    if T < 7:
        wl = T if T % 2 == 1 else T - 1
    else:
        wl = window
    if wl < 3:
        wl = 3

    y_smooth = savgol_filter(y_pos, window_length=wl, polyorder=min(3, wl - 1))
    y_vel = np.gradient(y_smooth, 1.0 / fps)

    # Foot plant = minimum vertical velocity in search window (most negative = descending fastest)
    search_vel = y_vel[start:end]
    local_idx = int(np.argmin(search_vel))
    return start + local_idx


def detect_mer(
    er_angles: NDArray[np.float64],
    foot_plant: int,
) -> int:
    """Detect maximum external rotation frame.

    Args:
        er_angles: (T,) shoulder external rotation angles in degrees.
        foot_plant: foot plant frame (search starts here).

    Returns:
        Frame index of peak external rotation.
    """
    search = er_angles[foot_plant:]
    if len(search) == 0:
        return foot_plant
    local_idx = int(np.argmax(search))
    return foot_plant + local_idx


def detect_release(
    wrist_positions: NDArray[np.float64],
    mer: int,
    fps: float = 30.0,
    window: int = 7,
) -> int:
    """Detect ball release frame.

    Strategy: peak forward-direction velocity of the throwing wrist after MER.

    Args:
        wrist_positions: (T, 3) throwing-wrist positions.
        mer: MER frame (search starts here).
        fps: frames per second.

    Returns:
        Frame index of release.
    """
    T = wrist_positions.shape[0]
    if mer >= T - 1:
        return T - 1

    wl = window if T >= window else (T if T % 2 == 1 else T - 1)
    if wl < 3:
        wl = 3

    vel_mag = _smooth_velocity(wrist_positions, fps=fps, window=wl)
    search = vel_mag[mer:]
    if len(search) == 0:
        return mer
    local_idx = int(np.argmax(search))
    return mer + local_idx


def detect_phases(
    joints_mhr70: NDArray[np.float64],
    er_angles: NDArray[np.float64],
    stride_ankle_idx: int,
    throw_wrist_idx: int,
    fps: float = 30.0,
) -> PitchPhases:
    """Run full phase detection pipeline.

    Args:
        joints_mhr70: (T, 70, 3) MHR70 joint positions.
        er_angles: (T,) precomputed shoulder external rotation angles.
        stride_ankle_idx: MHR70 index of the stride (front) foot ankle.
        throw_wrist_idx: MHR70 index of the throwing-hand wrist.
        fps: video frame rate.

    Returns:
        PitchPhases with foot_plant, mer, release frame indices.
    """
    T = joints_mhr70.shape[0]
    foot_plant = detect_foot_plant(joints_mhr70[:, stride_ankle_idx, :], fps=fps)
    mer = detect_mer(er_angles, foot_plant)
    release = detect_release(joints_mhr70[:, throw_wrist_idx, :], mer=mer, fps=fps)

    # Clamp to valid range
    foot_plant = max(0, min(foot_plant, T - 1))
    mer = max(foot_plant, min(mer, T - 1))
    release = max(mer, min(release, T - 1))

    return PitchPhases(
        foot_plant=foot_plant,
        mer=mer,
        release=release,
        total_frames=T,
    )
