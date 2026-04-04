"""FeatureExtractor: biomechanical feature extraction from SAM 3D Body output."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.signal import savgol_filter

from backend.app.pipeline.angles import (
    elbow_flexion,
    hip_flexion,
    knee_flexion,
    shoulder_abduction,
    shoulder_external_rotation,
    trunk_rotation,
    trunk_tilt,
)
from backend.app.pipeline.joint_map import MHR70, get_throwing_joints
from backend.app.pipeline.kinetics import KineticChainTiming, compute_kinetic_chain
from backend.app.pipeline.phases import PitchPhases, detect_phases
from backend.app.models.pitch import PitchData


@dataclass
class BiomechFeatures:
    """All extracted biomechanical features for one pitch."""

    # --- Joint angle time series (degrees) ---
    elbow_flexion: NDArray[np.float64]          # (T,)
    shoulder_abduction: NDArray[np.float64]     # (T,)
    shoulder_er: NDArray[np.float64]            # (T,) external rotation
    hip_flexion: NDArray[np.float64]            # (T,)
    knee_flexion: NDArray[np.float64]           # (T,)
    trunk_tilt: NDArray[np.float64]             # (T,)
    trunk_rotation: NDArray[np.float64]         # (T,)

    # --- Angular velocity time series (deg/s) ---
    elbow_flexion_vel: NDArray[np.float64]      # (T,)
    shoulder_er_vel: NDArray[np.float64]        # (T,)
    trunk_rotation_vel: NDArray[np.float64]     # (T,)

    # --- Scalar summary features ---
    max_shoulder_er_deg: float                  # normative peak ~170 deg
    hip_shoulder_sep_deg: float                 # normative 35-60 deg
    stride_length_normalized: float            # stride / pitcher height
    release_point: NDArray[np.float64]          # (3,) wrist position at release

    # --- Phase info ---
    phases: PitchPhases

    # --- Kinetic chain ---
    kinetic_chain: KineticChainTiming

    # --- Phase boundary export (for mesh export / viz) ---
    phase_boundaries: dict[str, int] = field(default_factory=dict)


def _angular_velocity(
    angles: NDArray[np.float64],
    fps: float = 30.0,
    window: int = 7,
) -> NDArray[np.float64]:
    """First derivative of angle sequence via Savitzky-Golay filter.

    Returns deg/s.
    """
    T = angles.shape[0]
    wl = window if T >= window else (T if T % 2 == 1 else max(3, T - 1))
    polyorder = min(3, wl - 1)
    if T >= wl:
        smoothed = savgol_filter(angles, window_length=wl, polyorder=polyorder)
    else:
        smoothed = angles
    return np.gradient(smoothed, 1.0 / fps)


def _hip_shoulder_separation(
    joints: NDArray[np.float64],
    left_hip_idx: int,
    right_hip_idx: int,
    left_shoulder_idx: int,
    right_shoulder_idx: int,
    frame: int,
) -> float:
    """Hip-shoulder separation at a given frame.

    Computed as the angle between the hip axis and shoulder axis
    projected onto the horizontal (xz) plane.

    Returns degrees.
    """
    hip_vec = joints[frame, right_hip_idx] - joints[frame, left_hip_idx]
    sho_vec = joints[frame, right_shoulder_idx] - joints[frame, left_shoulder_idx]
    # project onto horizontal plane
    hip_h = np.array([hip_vec[0], 0.0, hip_vec[2]])
    sho_h = np.array([sho_vec[0], 0.0, sho_vec[2]])
    nh = np.linalg.norm(hip_h)
    ns = np.linalg.norm(sho_h)
    if nh < 1e-9 or ns < 1e-9:
        return 0.0
    cos_a = np.clip(np.dot(hip_h / nh, sho_h / ns), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def _stride_length(
    joints: NDArray[np.float64],
    stride_ankle_idx: int,
    pivot_ankle_idx: int,
    foot_plant_frame: int,
    pitcher_height_m: float | None,
) -> float:
    """3D distance between ankles at foot plant, normalized by pitcher height.

    If pitcher_height_m is None, returns raw meters.
    """
    stride_pos = joints[foot_plant_frame, stride_ankle_idx]
    pivot_pos = joints[foot_plant_frame, pivot_ankle_idx]
    dist = float(np.linalg.norm(stride_pos - pivot_pos))
    if pitcher_height_m and pitcher_height_m > 0.1:
        return dist / pitcher_height_m
    return dist


class FeatureExtractor:
    """Extract biomechanical features from a PitchData object.

    Usage:
        extractor = FeatureExtractor(fps=30.0, handedness="right")
        features = extractor.extract(pitch_data)
    """

    def __init__(
        self,
        fps: float = 30.0,
        handedness: str = "right",
        pitcher_height_m: float | None = None,
        savgol_window: int = 7,
    ) -> None:
        self.fps = fps
        self.handedness = handedness
        self.pitcher_height_m = pitcher_height_m
        self.savgol_window = savgol_window
        self._joints = get_throwing_joints(handedness)

    def extract(self, pitch: PitchData) -> BiomechFeatures:
        """Run full feature extraction pipeline on a PitchData object.

        Uses joints_mhr70 if available, otherwise raises ValueError.
        """
        if pitch.joints_mhr70 is None:
            raise ValueError("joints_mhr70 is required for feature extraction")

        joints = np.array(pitch.joints_mhr70, dtype=np.float64)  # (T, 70, 3)
        return self.extract_from_array(joints)

    def extract_from_array(
        self,
        joints: NDArray[np.float64],
    ) -> BiomechFeatures:
        """Run feature extraction from a raw (T, 70, 3) joint array."""
        j = self._joints

        # --- Joint angles ---
        elbow_flex = elbow_flexion(
            joints, j["throw_shoulder"], j["throw_elbow"], j["throw_wrist"]
        )
        sho_abd = shoulder_abduction(
            joints, j["throw_elbow"], j["throw_shoulder"], j["glove_shoulder"]
        )
        sho_er = shoulder_external_rotation(
            joints,
            j["throw_elbow"],
            j["throw_shoulder"],
            j["glove_shoulder"],
            j["throw_wrist"],
        )
        hip_flex = hip_flexion(
            joints, j["stride_knee"], j["throw_hip"], j["glove_hip"]
        )
        knee_flex = knee_flexion(
            joints, j["throw_hip"], j["stride_knee"], j["stride_ankle"]
        )
        t_tilt = trunk_tilt(
            joints,
            MHR70["left_shoulder"],
            MHR70["right_shoulder"],
            MHR70["left_hip"],
            MHR70["right_hip"],
        )
        t_rot = trunk_rotation(
            joints,
            MHR70["left_shoulder"],
            MHR70["right_shoulder"],
            MHR70["left_hip"],
            MHR70["right_hip"],
        )

        # --- Angular velocities ---
        elbow_vel = _angular_velocity(elbow_flex, self.fps, self.savgol_window)
        sho_er_vel = _angular_velocity(sho_er, self.fps, self.savgol_window)
        t_rot_vel = _angular_velocity(t_rot, self.fps, self.savgol_window)

        # --- Phase detection ---
        phases = detect_phases(
            joints_mhr70=joints,
            er_angles=sho_er,
            stride_ankle_idx=j["stride_ankle"],
            throw_wrist_idx=j["throw_wrist"],
            fps=self.fps,
        )

        # --- Scalar features ---
        max_er = float(np.max(sho_er))

        hip_sho_sep = _hip_shoulder_separation(
            joints,
            MHR70["left_hip"],
            MHR70["right_hip"],
            MHR70["left_shoulder"],
            MHR70["right_shoulder"],
            frame=phases.foot_plant,
        )

        stride = _stride_length(
            joints,
            j["stride_ankle"],
            j["pivot_ankle"],
            phases.foot_plant,
            self.pitcher_height_m,
        )

        release_point = joints[phases.release, j["throw_wrist"]].copy()

        # --- Kinetic chain ---
        kinetics = compute_kinetic_chain(
            joints_mhr70=joints,
            left_hip_idx=MHR70["left_hip"],
            right_hip_idx=MHR70["right_hip"],
            left_shoulder_idx=MHR70["left_shoulder"],
            right_shoulder_idx=MHR70["right_shoulder"],
            throw_shoulder_idx=j["throw_shoulder"],
            throw_elbow_idx=j["throw_elbow"],
            throw_wrist_idx=j["throw_wrist"],
            foot_plant_frame=phases.foot_plant,
            fps=self.fps,
            search_end_frame=phases.release + 1,
        )

        return BiomechFeatures(
            elbow_flexion=elbow_flex,
            shoulder_abduction=sho_abd,
            shoulder_er=sho_er,
            hip_flexion=hip_flex,
            knee_flexion=knee_flex,
            trunk_tilt=t_tilt,
            trunk_rotation=t_rot,
            elbow_flexion_vel=elbow_vel,
            shoulder_er_vel=sho_er_vel,
            trunk_rotation_vel=t_rot_vel,
            max_shoulder_er_deg=max_er,
            hip_shoulder_sep_deg=hip_sho_sep,
            stride_length_normalized=stride,
            release_point=release_point,
            phases=phases,
            kinetic_chain=kinetics,
            phase_boundaries={
                "foot_plant": phases.foot_plant,
                "mer": phases.mer,
                "release": phases.release,
            },
        )
