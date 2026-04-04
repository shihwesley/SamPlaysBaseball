"""Tests for feature extraction pipeline.

Uses synthetic joint data — no SAM 3D Body output required.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.app.pipeline.angles import (
    angle_between,
    angle_sequence,
    elbow_flexion,
    shoulder_external_rotation,
    trunk_rotation,
    trunk_tilt,
)
from backend.app.pipeline.joint_map import MHR70, get_throwing_joints
from backend.app.pipeline.kinetics import compute_kinetic_chain
from backend.app.pipeline.phases import PitchPhases, detect_foot_plant, detect_mer, detect_phases, detect_release
from backend.app.pipeline.alignment import align_to_event, normalize_time_axis
from backend.app.pipeline.features import FeatureExtractor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_joints(T: int = 60, J: int = 70, seed: int = 0) -> np.ndarray:
    """Synthetic (T, J, 3) joint array with mild random motion."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 0.3, (J, 3))
    joints = np.tile(base, (T, 1, 1)).astype(np.float64)
    # Add smooth time-varying motion to a few joints
    t = np.linspace(0, 2 * np.pi, T)
    joints[:, MHR70["right_wrist"], 0] += np.sin(t) * 0.4
    joints[:, MHR70["right_wrist"], 1] += np.cos(t) * 0.2
    joints[:, MHR70["left_ankle"], 1] -= np.linspace(0, 0.5, T)  # ankle descends
    return joints


# ---------------------------------------------------------------------------
# Joint map
# ---------------------------------------------------------------------------

class TestJointMap:
    def test_mhr70_has_required_keys(self):
        required = ["left_shoulder", "right_shoulder", "left_hip", "right_hip",
                    "left_wrist", "right_wrist", "left_ankle", "right_ankle"]
        for k in required:
            assert k in MHR70, f"Missing key: {k}"

    def test_get_throwing_joints_right(self):
        j = get_throwing_joints("right")
        assert j["throw_shoulder"] == MHR70["right_shoulder"]
        assert j["throw_wrist"] == MHR70["right_wrist"]

    def test_get_throwing_joints_left(self):
        j = get_throwing_joints("left")
        assert j["throw_shoulder"] == MHR70["left_shoulder"]
        assert j["throw_wrist"] == MHR70["left_wrist"]


# ---------------------------------------------------------------------------
# Angles
# ---------------------------------------------------------------------------

class TestAngles:
    def test_angle_between_right_angle(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 0.0])
        c = np.array([0.0, 1.0, 0.0])
        angle = angle_between(a, b, c)
        assert abs(angle - 90.0) < 1e-5

    def test_angle_between_straight(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 0.0])
        c = np.array([-1.0, 0.0, 0.0])
        angle = angle_between(a, b, c)
        assert abs(angle - 180.0) < 1e-5

    def test_angle_between_zero_vectors(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 0.0])
        c = np.array([1.0, 0.0, 0.0])
        angle = angle_between(a, b, c)
        assert angle == 0.0

    def test_angle_sequence_shape(self):
        joints = make_joints(T=30)
        seq = angle_sequence(joints, MHR70["right_shoulder"], MHR70["right_elbow"], MHR70["right_wrist"])
        assert seq.shape == (30,)
        assert np.all(seq >= 0) and np.all(seq <= 180)

    def test_elbow_flexion_range(self):
        joints = make_joints(T=40)
        ef = elbow_flexion(joints, MHR70["right_shoulder"], MHR70["right_elbow"], MHR70["right_wrist"])
        assert ef.shape == (40,)
        assert np.all(ef >= 0)
        assert np.all(ef <= 180)

    def test_trunk_tilt_shape(self):
        joints = make_joints(T=50)
        tt = trunk_tilt(joints, MHR70["left_shoulder"], MHR70["right_shoulder"],
                        MHR70["left_hip"], MHR70["right_hip"])
        assert tt.shape == (50,)

    def test_trunk_rotation_shape(self):
        joints = make_joints(T=50)
        tr = trunk_rotation(joints, MHR70["left_shoulder"], MHR70["right_shoulder"],
                            MHR70["left_hip"], MHR70["right_hip"])
        assert tr.shape == (50,)
        # Rotation should be between -180 and 180
        assert np.all(tr >= -180) and np.all(tr <= 180)

    def test_shoulder_er_shape(self):
        joints = make_joints(T=40)
        er = shoulder_external_rotation(
            joints,
            MHR70["right_elbow"],
            MHR70["right_shoulder"],
            MHR70["left_shoulder"],
            MHR70["right_wrist"],
        )
        assert er.shape == (40,)
        assert np.all(er >= 0)


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

class TestPhases:
    def test_detect_foot_plant_returns_valid_frame(self):
        joints = make_joints(T=60)
        fp = detect_foot_plant(joints[:, MHR70["left_ankle"], :], fps=30.0)
        assert 0 <= fp < 60

    def test_detect_mer_after_foot_plant(self):
        er = np.zeros(60)
        er[40] = 170.0  # peak at frame 40
        mer = detect_mer(er, foot_plant=10)
        assert mer == 40

    def test_detect_release_after_mer(self):
        joints = make_joints(T=60)
        release = detect_release(joints[:, MHR70["right_wrist"], :], mer=30, fps=30.0)
        assert release >= 30

    def test_detect_phases_valid_ordering(self):
        joints = make_joints(T=60)
        er = np.zeros(60)
        er[35] = 170.0
        phases = detect_phases(
            joints_mhr70=joints,
            er_angles=er,
            stride_ankle_idx=MHR70["left_ankle"],
            throw_wrist_idx=MHR70["right_wrist"],
            fps=30.0,
        )
        assert isinstance(phases, PitchPhases)
        assert phases.foot_plant <= phases.mer
        assert phases.mer <= phases.release
        assert phases.total_frames == 60

    def test_phases_valid_property(self):
        phases = PitchPhases(foot_plant=10, mer=30, release=45, total_frames=60)
        assert phases.valid

    def test_phases_invalid_order(self):
        phases = PitchPhases(foot_plant=40, mer=30, release=20, total_frames=60)
        assert not phases.valid

    def test_detect_phases_short_sequence(self):
        """Should not crash on very short sequences."""
        joints = make_joints(T=10)
        er = np.zeros(10)
        er[7] = 150.0
        phases = detect_phases(
            joints_mhr70=joints,
            er_angles=er,
            stride_ankle_idx=MHR70["left_ankle"],
            throw_wrist_idx=MHR70["right_wrist"],
            fps=30.0,
        )
        assert phases.total_frames == 10


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

class TestAlignment:
    def test_align_to_event_shape(self):
        joints = make_joints(T=60)
        phases = PitchPhases(foot_plant=15, mer=35, release=50, total_frames=60)
        aligned = align_to_event(joints, phases, anchor="mer", pre_frames=10, post_frames=10)
        assert aligned.shape == (21, 70, 3)

    def test_align_to_event_unknown_anchor_raises(self):
        joints = make_joints(T=60)
        phases = PitchPhases(foot_plant=15, mer=35, release=50, total_frames=60)
        with pytest.raises(ValueError):
            align_to_event(joints, phases, anchor="invalid")

    def test_normalize_time_axis_shape(self):
        joints = make_joints(T=60)
        phases = PitchPhases(foot_plant=15, mer=35, release=50, total_frames=60)
        resampled = normalize_time_axis(joints, phases, n_samples=99)
        assert resampled.shape[0] == 99
        assert resampled.shape[1] == 70
        assert resampled.shape[2] == 3


# ---------------------------------------------------------------------------
# Kinetics
# ---------------------------------------------------------------------------

class TestKinetics:
    def test_kinetic_chain_returns_valid_frames(self):
        joints = make_joints(T=60)
        kt = compute_kinetic_chain(
            joints_mhr70=joints,
            left_hip_idx=MHR70["left_hip"],
            right_hip_idx=MHR70["right_hip"],
            left_shoulder_idx=MHR70["left_shoulder"],
            right_shoulder_idx=MHR70["right_shoulder"],
            throw_shoulder_idx=MHR70["right_shoulder"],
            throw_elbow_idx=MHR70["right_elbow"],
            throw_wrist_idx=MHR70["right_wrist"],
            foot_plant_frame=15,
            fps=30.0,
            search_end_frame=55,
        )
        for frame in [kt.pelvis_peak_frame, kt.trunk_peak_frame,
                      kt.shoulder_peak_frame, kt.elbow_peak_frame,
                      kt.wrist_peak_frame]:
            assert 0 <= frame < 60

    def test_kinetic_chain_as_dict(self):
        joints = make_joints(T=60)
        kt = compute_kinetic_chain(
            joints_mhr70=joints,
            left_hip_idx=MHR70["left_hip"],
            right_hip_idx=MHR70["right_hip"],
            left_shoulder_idx=MHR70["left_shoulder"],
            right_shoulder_idx=MHR70["right_shoulder"],
            throw_shoulder_idx=MHR70["right_shoulder"],
            throw_elbow_idx=MHR70["right_elbow"],
            throw_wrist_idx=MHR70["right_wrist"],
            foot_plant_frame=10,
            fps=30.0,
        )
        d = kt.as_dict()
        assert set(d.keys()) == {
            "pelvis_peak_frame", "trunk_peak_frame", "shoulder_peak_frame",
            "elbow_peak_frame", "wrist_peak_frame",
        }


# ---------------------------------------------------------------------------
# FeatureExtractor
# ---------------------------------------------------------------------------

class TestFeatureExtractor:
    def test_extract_from_array_returns_features(self):
        extractor = FeatureExtractor(fps=30.0, handedness="right", pitcher_height_m=1.88)
        joints = make_joints(T=60)
        features = extractor.extract_from_array(joints)

        assert features.elbow_flexion.shape == (60,)
        assert features.shoulder_er.shape == (60,)
        assert features.phases.total_frames == 60
        assert features.release_point.shape == (3,)
        assert isinstance(features.max_shoulder_er_deg, float)
        assert isinstance(features.hip_shoulder_sep_deg, float)
        assert isinstance(features.stride_length_normalized, float)

    def test_extract_from_array_left_handed(self):
        extractor = FeatureExtractor(fps=30.0, handedness="left")
        joints = make_joints(T=50)
        features = extractor.extract_from_array(joints)
        assert features.phases.total_frames == 50

    def test_extract_from_array_no_height(self):
        extractor = FeatureExtractor(fps=30.0, handedness="right", pitcher_height_m=None)
        joints = make_joints(T=40)
        features = extractor.extract_from_array(joints)
        assert isinstance(features.stride_length_normalized, float)

    def test_phase_boundaries_exported(self):
        extractor = FeatureExtractor(fps=30.0)
        joints = make_joints(T=60)
        features = extractor.extract_from_array(joints)
        assert "foot_plant" in features.phase_boundaries
        assert "mer" in features.phase_boundaries
        assert "release" in features.phase_boundaries

    def test_extract_raises_without_mhr70(self):
        from datetime import datetime
        from backend.app.models.pitch import PitchData, PitchMetadata
        extractor = FeatureExtractor()
        meta = PitchMetadata(
            pitcher_id="12345",
            game_date=datetime(2023, 6, 1),
            inning=1,
            pitch_number=1,
            pitch_type="FF",
        )
        # Create PitchData without joints_mhr70
        joints_np = np.zeros((10, 127, 3))
        pitch = PitchData.from_numpy(
            metadata=meta,
            joints=joints_np,
            joints_mhr70=None,
            pose_params=np.zeros((10, 136)),
            shape_params=np.zeros(45),
        )
        with pytest.raises(ValueError, match="joints_mhr70"):
            extractor.extract(pitch)

    def test_extract_with_mhr70(self):
        from datetime import datetime
        from backend.app.models.pitch import PitchData, PitchMetadata
        extractor = FeatureExtractor()
        meta = PitchMetadata(
            pitcher_id="12345",
            game_date=datetime(2023, 6, 1),
            inning=1,
            pitch_number=1,
            pitch_type="FF",
        )
        joints_np = np.zeros((30, 127, 3))
        joints_mhr70 = make_joints(T=30)
        pitch = PitchData.from_numpy(
            metadata=meta,
            joints=joints_np,
            joints_mhr70=joints_mhr70,
            pose_params=np.zeros((30, 136)),
            shape_params=np.zeros(45),
        )
        features = extractor.extract(pitch)
        assert features.phases.total_frames == 30
