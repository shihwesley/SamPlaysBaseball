"""Tests for ReportGenerator and report templates."""

from __future__ import annotations

import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.app.models.baseline import JointStats, PitcherBaseline, PitchTypeBaseline, PoseParamStats
from backend.app.models.pitch import PitchData, PitchMetadata
from backend.app.models.storage import StorageLayer
from backend.app.reports.generator import ReportGenerator, ScoutingReport
from backend.app.reports import templates


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_storage(tmp_path: Path) -> StorageLayer:
    db = tmp_path / "test.db"
    parquet = tmp_path / "parquet"
    storage = StorageLayer(db_path=db, parquet_dir=parquet)

    # Add pitcher baseline
    joint_stats = [JointStats(mean=[0.0, 0.0, 0.0], std=[0.1, 0.1, 0.1]) for _ in range(127)]
    pose_stats = PoseParamStats(mean=[0.0] * 136, std=[0.1] * 136)
    pt_baseline = PitchTypeBaseline(
        pitch_type="FF",
        sample_count=15,
        joint_stats=joint_stats,
        pose_param_stats=pose_stats,
        shape_params_mean=[0.0] * 45,
    )
    baseline = PitcherBaseline(
        pitcher_id="p1",
        pitcher_name="Test Pitcher",
        handedness="R",
        by_pitch_type={"FF": pt_baseline},
    )
    storage.save_baseline(baseline)

    # Add one pitch
    meta = PitchMetadata(
        pitch_id="pitch-1",
        pitcher_id="p1",
        game_date=datetime(2024, 5, 15),
        inning=2,
        pitch_number=10,
        pitch_type="FF",
        velocity_mph=93.5,
    )
    pitch = PitchData(
        metadata=meta,
        joints=[[[0.0, 0.0, 0.0]] * 127] * 3,
        joints_mhr70=[[[0.0, 0.0, 0.0]] * 70] * 3,
        pose_params=[[0.0] * 136] * 3,
        shape_params=[0.0] * 45,
    )
    storage.save_pitch(pitch)

    return storage


@pytest.fixture
def storage(tmp_path) -> StorageLayer:
    return make_storage(tmp_path)


# ---------------------------------------------------------------------------
# Template tests
# ---------------------------------------------------------------------------


def test_pitcher_profile_section_non_empty():
    result = templates.pitcher_profile_section(
        "p1", "Test Pitcher", ["FF", "SL"], {"FF": 20, "SL": 10}
    )
    assert result
    assert "Test Pitcher" in result
    assert "FF" in result


def test_tipping_section_no_tipping():
    result = templates.tipping_section([], 0.0, False)
    assert result
    assert "No significant" in result


def test_tipping_section_with_tipping():
    signals = [
        {"feature_name": "elbow_flex_mean", "pitch_type_a": "FF", "pitch_type_b": "SL", "separation_score": 0.82}
    ]
    result = templates.tipping_section(signals, 0.82, True)
    assert "0.82" in result
    assert "elbow_flex_mean" in result


def test_fatigue_section_not_fatigued():
    result = templates.fatigue_section([], 0.3, False)
    assert result
    assert "acceptable" in result


def test_fatigue_section_fatigued():
    markers = [{"metric_name": "arm_slot_drop", "pct_change": -12.5}]
    result = templates.fatigue_section(markers, 0.75, True)
    assert "0.75" in result
    assert "elevated" in result


def test_command_section():
    result = templates.command_section(0.82, {"release_x_deviation": 3.2, "release_z_deviation": 1.5})
    assert "0.82" in result
    assert "strong" in result


def test_arm_slot_section_no_data():
    result = templates.arm_slot_section(None, None, False)
    assert result
    assert "unavailable" in result


def test_arm_slot_section_with_drift():
    result = templates.arm_slot_section(2.5, 180.0, False)
    assert "2.5" in result


def test_timing_section():
    events = [{"event_name": "foot_plant", "frame": 10, "frame_delta": 3}]
    result = templates.timing_section(events, 0.7, True)
    assert "0.70" in result
    assert "late" in result


def test_injury_risk_section():
    factors = [("fatigue_mechanics_drift", 0.75), ("arm_slot_instability", 0.45)]
    result = templates.injury_risk_section(55.0, "yellow", factors)
    assert "55.0" in result
    assert "YELLOW" in result


def test_statcast_section_no_data():
    result = templates.statcast_section(None)
    assert result
    assert "not available" in result


# ---------------------------------------------------------------------------
# ReportGenerator tests
# ---------------------------------------------------------------------------


def test_generate_pitcher_report_no_llm(storage):
    gen = ReportGenerator(storage=storage, llm=None)
    report = gen.generate_pitcher_report("p1")

    assert isinstance(report, ScoutingReport)
    assert report.pitcher_id == "p1"
    assert report.pitcher_name == "Test Pitcher"
    assert report.report_type == "pitcher"
    assert "pitcher_profile" in report.sections
    assert report.narrative  # non-empty
    assert isinstance(report.recommendations, list)
    assert report.risk_level in ("green", "yellow", "orange", "red")


def test_generate_pitcher_report_required_fields(storage):
    gen = ReportGenerator(storage=storage, llm=None)
    report = gen.generate_pitcher_report("p1")

    # All required ScoutingReport fields
    assert report.generated_at is not None
    assert isinstance(report.sections, dict)
    assert isinstance(report.metadata, dict)


def test_generate_outing_report(storage):
    gen = ReportGenerator(storage=storage, llm=None)
    report = gen.generate_outing_report("p1", "2024-05-15")

    assert report.report_type == "outing"
    assert report.pitcher_id == "p1"
    assert "fatigue" in report.sections


def test_generate_pitch_type_report(storage):
    gen = ReportGenerator(storage=storage, llm=None)
    report = gen.generate_pitch_type_report("p1", "FF")

    assert report.report_type == "pitch_type"
    assert report.metadata.get("pitch_type") == "FF"


def test_narrative_fallback_without_llm(storage):
    """Without LLM, narrative should be template concatenation (non-empty)."""
    gen = ReportGenerator(storage=storage, llm=None)
    report = gen.generate_pitcher_report("p1")
    # Narrative is the concatenation of sections
    assert len(report.narrative) > 0
    for section_text in report.sections.values():
        assert section_text in report.narrative


def test_llm_narrative_called_when_available(storage):
    """When LLM is provided, generate_narrative should be called."""
    mock_llm = MagicMock()
    mock_llm.generate_narrative.return_value = "Mock narrative from LLM."
    mock_llm.generate_recommendations.return_value = ["Do this", "Try that"]

    gen = ReportGenerator(storage=storage, llm=mock_llm)
    report = gen.generate_pitcher_report("p1")

    mock_llm.generate_narrative.assert_called_once()
    assert report.narrative == "Mock narrative from LLM."


def test_llm_failure_falls_back_to_template(storage):
    """If LLM raises, narrative falls back to template concatenation."""
    mock_llm = MagicMock()
    mock_llm.generate_narrative.side_effect = RuntimeError("API error")
    mock_llm.generate_recommendations.side_effect = RuntimeError("API error")

    gen = ReportGenerator(storage=storage, llm=mock_llm)
    report = gen.generate_pitcher_report("p1")

    # Fallback: narrative is non-empty template text
    assert len(report.narrative) > 0


def test_llm_mock_init():
    """Test LLMReportGenerator initialisation with mocked Anthropic client."""
    with patch("backend.app.reports.llm.Anthropic") as MockAnthropicCls:
        mock_client = MagicMock()
        MockAnthropicCls.return_value = mock_client

        from backend.app.reports.llm import LLMReportGenerator

        llm = LLMReportGenerator(api_key="test-key", model="claude-opus-4-5")
        assert llm.model == "claude-opus-4-5"
        MockAnthropicCls.assert_called_once_with(api_key="test-key")


def test_llm_generate_narrative_calls_api(storage):
    """generate_narrative calls Anthropic messages.create and returns text."""
    with patch("backend.app.reports.llm.Anthropic") as MockAnthropicCls:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Scouting narrative text.")]
        mock_client.messages.create.return_value = mock_response
        MockAnthropicCls.return_value = mock_client

        from backend.app.reports.llm import LLMReportGenerator

        llm = LLMReportGenerator(api_key="test-key")
        result = llm.generate_narrative({"profile": "Test"}, "Test Pitcher")

        assert result == "Scouting narrative text."
        mock_client.messages.create.assert_called_once()
