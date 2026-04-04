"""Tests for data model behavior.

Tests behavior, not type-system guarantees.
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from app.models import (
    ArmSlotDriftResult,
    BaselineComparisonResult,
    CommandAnalysisResult,
    FatigueTrackingResult,
    PitchData,
    PitchMetadata,
    PitcherBaseline,
    StorageLayer,
    TimingAnalysisResult,
    TippingDetectionResult,
)
from app.models.baseline import JointStats, PitchTypeBaseline, PoseParamStats
from app.models.analysis import (
    AnyAnalysisResult,
    JointDeviation,
    FatigueMarker,
    TimingEvent,
    TipSignal,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_metadata(pitch_id: str = "test-pitch-1", pitch_type: str = "FF") -> PitchMetadata:
    return PitchMetadata(
        pitch_id=pitch_id,
        pitcher_id="pitcher-1",
        game_date=datetime(2024, 7, 4, 19, 0, 0),
        inning=3,
        pitch_number=42,
        pitch_type=pitch_type,
        velocity_mph=95.2,
        spin_rate_rpm=2350,
    )


def make_pitch_numpy(T: int = 10):
    joints = np.random.rand(T, 127, 3).astype(np.float32)
    joints_mhr70 = np.random.rand(T, 70, 3).astype(np.float32)
    pose_params = np.random.rand(T, 136).astype(np.float32)
    shape_params = np.random.rand(45).astype(np.float32)
    return joints, joints_mhr70, pose_params, shape_params


def make_pitch(T: int = 10, pitch_id: str = "test-pitch-1") -> PitchData:
    meta = make_metadata(pitch_id=pitch_id)
    joints, joints_mhr70, pose_params, shape_params = make_pitch_numpy(T)
    return PitchData.from_numpy(meta, joints, pose_params, shape_params, joints_mhr70)


def make_storage(tmp_path: Path) -> StorageLayer:
    return StorageLayer(
        db_path=tmp_path / "test.db",
        parquet_dir=tmp_path / "parquet",
    )


# ---------------------------------------------------------------------------
# PitchMetadata
# ---------------------------------------------------------------------------


def test_pitch_metadata_defaults_generate_unique_ids():
    m1 = make_metadata()
    # Using explicit pitch_id so uniqueness is controlled by caller
    m2 = PitchMetadata(
        pitcher_id="p",
        game_date=datetime(2024, 1, 1),
        inning=1,
        pitch_number=1,
        pitch_type="FF",
    )
    m3 = PitchMetadata(
        pitcher_id="p",
        game_date=datetime(2024, 1, 1),
        inning=1,
        pitch_number=2,
        pitch_type="FF",
    )
    assert m2.pitch_id != m3.pitch_id


# ---------------------------------------------------------------------------
# PitchData
# ---------------------------------------------------------------------------


def test_pitch_data_num_frames_set_from_joints():
    pitch = make_pitch(T=15)
    assert pitch.num_frames == 15


def test_pitch_data_array_shapes():
    T = 12
    pitch = make_pitch(T=T)
    assert pitch.joints_array().shape == (T, 127, 3)
    assert pitch.joints_mhr70_array().shape == (T, 70, 3)
    assert pitch.pose_params_array().shape == (T, 136)
    assert pitch.shape_params_array().shape == (45,)


def test_pitch_data_json_roundtrip():
    pitch = make_pitch()
    data = pitch.model_dump_json()
    restored = PitchData.model_validate_json(data)
    assert restored.metadata.pitch_id == pitch.metadata.pitch_id
    assert restored.num_frames == pitch.num_frames
    np.testing.assert_allclose(restored.joints_array(), pitch.joints_array(), atol=1e-5)


def test_pitch_data_without_mhr70():
    meta = make_metadata()
    joints, _, pose_params, shape_params = make_pitch_numpy()
    pitch = PitchData.from_numpy(meta, joints, pose_params, shape_params)
    assert pitch.joints_mhr70 is None
    assert pitch.joints_mhr70_array() is None
    assert pitch.skeleton_type == "full"


# ---------------------------------------------------------------------------
# PitcherBaseline
# ---------------------------------------------------------------------------


def _make_joint_stats(n: int) -> list[JointStats]:
    return [JointStats(mean=[0.0, 0.0, 0.0], std=[0.1, 0.1, 0.1]) for _ in range(n)]


def make_baseline(pitcher_id: str = "pitcher-1") -> PitcherBaseline:
    pt_baseline = PitchTypeBaseline(
        pitch_type="FF",
        sample_count=50,
        joint_stats=_make_joint_stats(127),
        joint_stats_mhr70=_make_joint_stats(70),
        pose_param_stats=PoseParamStats(mean=[0.0] * 136, std=[0.01] * 136),
        shape_params_mean=[0.0] * 45,
    )
    return PitcherBaseline(
        pitcher_id=pitcher_id,
        pitcher_name="Shohei Ohtani",
        handedness="R",
        by_pitch_type={"FF": pt_baseline},
        shape_params_mean=[0.0] * 45,
        shape_params_std=[0.01] * 45,
    )


def test_baseline_get_existing_pitch_type():
    baseline = make_baseline()
    pt = baseline.get_baseline("FF")
    assert pt is not None
    assert pt.sample_count == 50


def test_baseline_get_missing_pitch_type_returns_none():
    baseline = make_baseline()
    assert baseline.get_baseline("SL") is None


def test_baseline_json_roundtrip():
    baseline = make_baseline()
    restored = PitcherBaseline.model_validate_json(baseline.model_dump_json())
    assert restored.pitcher_id == baseline.pitcher_id
    ff = restored.get_baseline("FF")
    assert ff is not None
    assert len(ff.joint_stats) == 127


# ---------------------------------------------------------------------------
# Analysis results
# ---------------------------------------------------------------------------


def test_baseline_comparison_result():
    result = BaselineComparisonResult(
        pitch_id="p1",
        pitcher_id="pitcher-1",
        pitch_type="FF",
        overall_z_score=1.8,
        top_deviations=[
            JointDeviation(joint_index=5, joint_name="right_elbow", deviation_mm=12.3, z_score=2.1)
        ],
        is_outlier=False,
    )
    assert result.module == "baseline-comparison"
    data = json.loads(result.model_dump_json())
    assert data["module"] == "baseline-comparison"


def test_tipping_detection_result():
    result = TippingDetectionResult(
        pitch_id="p1",
        pitcher_id="pitcher-1",
        tip_signals=[
            TipSignal(
                feature_name="glove_position_at_set",
                pitch_type_a="FF",
                pitch_type_b="CH",
                separation_score=0.82,
            )
        ],
        max_separation_score=0.82,
        is_tipping=True,
    )
    assert result.module == "tipping-detection"
    assert result.is_tipping is True


def test_fatigue_tracking_result():
    result = FatigueTrackingResult(
        pitch_id="p1",
        pitcher_id="pitcher-1",
        pitch_number_in_game=85,
        markers=[
            FatigueMarker(
                metric_name="arm_slot_deg",
                value=2.1,
                baseline_value=5.3,
                pct_change=-0.60,
            )
        ],
        fatigue_score=0.7,
        is_fatigued=True,
    )
    assert result.module == "fatigue-tracking"


def test_command_analysis_result():
    result = CommandAnalysisResult(
        pitch_id="p1",
        pitcher_id="pitcher-1",
        pitch_type="FF",
        plate_x=0.3,
        plate_z=2.1,
        command_score=0.85,
        zone=2,
    )
    assert result.module == "command-analysis"


def test_arm_slot_drift_result():
    result = ArmSlotDriftResult(
        pitch_id="p1",
        pitcher_id="pitcher-1",
        arm_slot_degrees=45.2,
        baseline_arm_slot_degrees=48.0,
        drift_degrees=-2.8,
        is_significant_drift=False,
    )
    assert result.module == "arm-slot-drift"


def test_timing_analysis_result():
    result = TimingAnalysisResult(
        pitch_id="p1",
        pitcher_id="pitcher-1",
        pitch_type="FF",
        events=[
            TimingEvent(event_name="foot_plant", frame=12, baseline_frame=10, frame_delta=2)
        ],
        timing_score=0.9,
    )
    assert result.module == "timing-analysis"


def test_analysis_result_confidence_bounds():
    with pytest.raises(Exception):
        BaselineComparisonResult(
            pitch_id="p1",
            pitcher_id="p1",
            pitch_type="FF",
            overall_z_score=1.0,
            confidence=1.5,  # invalid
        )


# ---------------------------------------------------------------------------
# StorageLayer
# ---------------------------------------------------------------------------


def test_storage_save_and_load_pitch(tmp_path):
    store = make_storage(tmp_path)
    pitch = make_pitch(T=8, pitch_id="pitch-save-load")
    store.save_pitch(pitch)

    loaded = store.load_pitch("pitch-save-load")
    assert loaded is not None
    assert loaded.metadata.pitch_id == "pitch-save-load"
    assert loaded.num_frames == 8
    np.testing.assert_allclose(loaded.joints_array(), pitch.joints_array(), atol=1e-5)
    np.testing.assert_allclose(
        loaded.joints_mhr70_array(), pitch.joints_mhr70_array(), atol=1e-5
    )
    np.testing.assert_allclose(
        loaded.pose_params_array(), pitch.pose_params_array(), atol=1e-5
    )
    np.testing.assert_allclose(
        loaded.shape_params_array(), pitch.shape_params_array(), atol=1e-5
    )


def test_storage_load_missing_pitch_returns_none(tmp_path):
    store = make_storage(tmp_path)
    assert store.load_pitch("nonexistent") is None


def test_storage_list_pitch_ids(tmp_path):
    store = make_storage(tmp_path)
    for i in range(3):
        pitch = make_pitch(pitch_id=f"pitch-{i}")
        # Ensure same pitcher_id
        pitch.metadata.pitcher_id = "pitcher-abc"
        store.save_pitch(pitch)

    ids = store.list_pitch_ids(pitcher_id="pitcher-abc")
    assert len(ids) == 3
    assert set(ids) == {"pitch-0", "pitch-1", "pitch-2"}


def test_storage_save_and_load_baseline(tmp_path):
    store = make_storage(tmp_path)
    baseline = make_baseline()
    store.save_baseline(baseline)

    loaded = store.load_baseline("pitcher-1")
    assert loaded is not None
    assert loaded.pitcher_name == "Shohei Ohtani"
    ff = loaded.get_baseline("FF")
    assert ff is not None
    assert ff.sample_count == 50


def test_storage_load_missing_baseline_returns_none(tmp_path):
    store = make_storage(tmp_path)
    assert store.load_baseline("nobody") is None


def test_storage_save_and_load_analysis(tmp_path):
    store = make_storage(tmp_path)
    result = BaselineComparisonResult(
        pitch_id="p1",
        pitcher_id="pitcher-1",
        pitch_type="FF",
        overall_z_score=2.3,
        is_outlier=True,
    )
    store.save_analysis(result)

    rows = store.load_analysis("p1", module="baseline-comparison")
    assert len(rows) == 1
    assert rows[0]["overall_z_score"] == pytest.approx(2.3)
    assert rows[0]["is_outlier"] is True


def test_storage_pitch_upsert(tmp_path):
    """Saving the same pitch_id twice replaces it."""
    store = make_storage(tmp_path)
    pitch = make_pitch(T=5, pitch_id="upsert-test")
    store.save_pitch(pitch)

    pitch2 = make_pitch(T=7, pitch_id="upsert-test")
    store.save_pitch(pitch2)

    loaded = store.load_pitch("upsert-test")
    assert loaded.num_frames == 7
