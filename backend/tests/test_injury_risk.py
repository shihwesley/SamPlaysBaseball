"""Tests for InjuryRiskCalculator.

All tests use synthetic data -- no real upstream objects are constructed.
Mock results are plain dicts that match the field names of the Pydantic models.
"""

from __future__ import annotations

import pytest

from backend.app.analysis.injury_risk import InjuryRiskCalculator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fatigue(score: float) -> dict:
    """Minimal FatigueTrackingResult-shaped dict."""
    return {"fatigue_score": score}


def _arm_slot(drift: float, cumulative: float | None = None) -> dict:
    """Minimal ArmSlotDriftResult-shaped dict."""
    return {"drift_degrees": drift, "cumulative_drift_degrees": cumulative}


def _timing(timing_score: float) -> dict:
    """Minimal TimingAnalysisResult-shaped dict."""
    return {"timing_score": timing_score}


def _zero_inputs():
    return dict(
        fatigue_result=_fatigue(0.0),
        arm_slot_result=_arm_slot(0.0),
        timing_result=_timing(1.0),
        pitch_count=0,
        days_since_last_outing=4,
        hip_shoulder_separation_angle=30.0,
    )


def _max_inputs():
    return dict(
        fatigue_result=_fatigue(1.0),
        arm_slot_result=_arm_slot(10.0),
        timing_result=_timing(0.0),
        pitch_count=100,
        days_since_last_outing=0,
        hip_shoulder_separation_angle=0.0,
    )


# ---------------------------------------------------------------------------
# Test 1: Zero risk
# ---------------------------------------------------------------------------

def test_zero_risk():
    calc = InjuryRiskCalculator()
    result = calc.calculate_risk(**_zero_inputs())
    assert result["risk_score"] == pytest.approx(0.0, abs=1.0)
    assert result["traffic_light"] == "green"


# ---------------------------------------------------------------------------
# Test 2: Maximum risk
# ---------------------------------------------------------------------------

def test_maximum_risk():
    calc = InjuryRiskCalculator()
    result = calc.calculate_risk(**_max_inputs())
    assert result["risk_score"] == pytest.approx(100.0, abs=1.0)
    assert result["traffic_light"] == "red"


# ---------------------------------------------------------------------------
# Test 3: Weight correctness — isolate fatigue factor
# ---------------------------------------------------------------------------

def test_fatigue_weight():
    """With only fatigue at 1.0 and all others at 0, score should be 0.30 * 100 = 30."""
    calc = InjuryRiskCalculator()
    result = calc.calculate_risk(
        fatigue_result=_fatigue(1.0),
        arm_slot_result=_arm_slot(0.0),
        timing_result=_timing(1.0),
        pitch_count=0,
        days_since_last_outing=4,
        hip_shoulder_separation_angle=30.0,
    )
    assert result["risk_score"] == pytest.approx(30.0, abs=0.5)


def test_kinetic_chain_weight():
    """Isolated kinetic chain disruption (timing_score=0) -> 0.25 * 100 = 25."""
    calc = InjuryRiskCalculator()
    result = calc.calculate_risk(
        fatigue_result=_fatigue(0.0),
        arm_slot_result=_arm_slot(0.0),
        timing_result=_timing(0.0),
        pitch_count=0,
        days_since_last_outing=4,
        hip_shoulder_separation_angle=30.0,
    )
    assert result["risk_score"] == pytest.approx(25.0, abs=0.5)


def test_arm_slot_weight():
    """Isolated arm slot at max drift (10 deg) -> 0.20 * 100 = 20."""
    calc = InjuryRiskCalculator()
    result = calc.calculate_risk(
        fatigue_result=_fatigue(0.0),
        arm_slot_result=_arm_slot(10.0),
        timing_result=_timing(1.0),
        pitch_count=0,
        days_since_last_outing=4,
        hip_shoulder_separation_angle=30.0,
    )
    assert result["risk_score"] == pytest.approx(20.0, abs=0.5)


# ---------------------------------------------------------------------------
# Test 4: Traffic light thresholds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("score,expected", [
    (0.0, "green"),
    (30.0, "green"),
    (31.0, "yellow"),
    (60.0, "yellow"),
    (61.0, "orange"),
    (80.0, "orange"),
    (81.0, "red"),
    (100.0, "red"),
])
def test_traffic_light(score, expected):
    calc = InjuryRiskCalculator()
    assert calc._traffic_light(score) == expected


# ---------------------------------------------------------------------------
# Test 5: Trend detection
# ---------------------------------------------------------------------------

def test_increasing_trend_detected():
    calc = InjuryRiskCalculator()
    pitcher = "test_pitcher"
    calc.update_trend(pitcher, 20.0)
    calc.update_trend(pitcher, 30.0)
    result = calc.update_trend(pitcher, 40.0)
    assert result["increasing_trend"] is True
    assert result["consecutive_increases"] >= 3


def test_no_trend_on_decrease():
    calc = InjuryRiskCalculator()
    pitcher = "test_pitcher_2"
    calc.update_trend(pitcher, 40.0)
    calc.update_trend(pitcher, 50.0)
    result = calc.update_trend(pitcher, 45.0)
    assert result["increasing_trend"] is False


def test_trend_resets_on_decrease():
    """A single decrease interrupts a prior streak; streak restarts from that point."""
    calc = InjuryRiskCalculator()
    pitcher = "test_pitcher_3"
    calc.update_trend(pitcher, 40.0)
    calc.update_trend(pitcher, 50.0)
    # Decrease here -- streak drops to 1
    result = calc.update_trend(pitcher, 45.0)
    assert result["increasing_trend"] is False
    assert result["consecutive_increases"] == 1


# ---------------------------------------------------------------------------
# Test 6: Workload normalization
# ---------------------------------------------------------------------------

def test_workload_100_pitches():
    calc = InjuryRiskCalculator()
    factor = calc._workload_factor(pitch_count=100, days_since_last_outing=4)
    # pitch component = 1.0, rest deficit = 0.0 -> 0.6 * 1.0 + 0.4 * 0.0 = 0.6
    assert factor == pytest.approx(0.6, abs=0.01)


def test_workload_4_days_rest():
    calc = InjuryRiskCalculator()
    factor = calc._workload_factor(pitch_count=0, days_since_last_outing=4)
    # pitch = 0, rest deficit = 0 -> 0.0
    assert factor == pytest.approx(0.0, abs=0.01)


def test_workload_0_days_rest():
    calc = InjuryRiskCalculator()
    factor = calc._workload_factor(pitch_count=0, days_since_last_outing=0)
    # pitch = 0, rest deficit = 1.0 -> 0.4
    assert factor == pytest.approx(0.4, abs=0.01)


def test_workload_exceeds_100_pitches():
    calc = InjuryRiskCalculator()
    factor = calc._workload_factor(pitch_count=120, days_since_last_outing=4)
    assert factor == pytest.approx(0.6, abs=0.01)  # clamped at 1.0 for pitches


# ---------------------------------------------------------------------------
# Test 7: Hip-shoulder separation deficit
# ---------------------------------------------------------------------------

def test_hip_shoulder_30_no_deficit():
    calc = InjuryRiskCalculator()
    factor = calc._hip_shoulder_factor(30.0)
    assert factor == pytest.approx(0.0, abs=0.001)


def test_hip_shoulder_0_full_deficit():
    calc = InjuryRiskCalculator()
    factor = calc._hip_shoulder_factor(0.0)
    assert factor == pytest.approx(1.0, abs=0.001)


def test_hip_shoulder_15_half_deficit():
    calc = InjuryRiskCalculator()
    factor = calc._hip_shoulder_factor(15.0)
    assert factor == pytest.approx(0.5, abs=0.001)


def test_hip_shoulder_above_30_no_deficit():
    calc = InjuryRiskCalculator()
    factor = calc._hip_shoulder_factor(45.0)
    assert factor == pytest.approx(0.0, abs=0.001)


# ---------------------------------------------------------------------------
# Test 8: Graceful degradation with None inputs
# ---------------------------------------------------------------------------

def test_none_fatigue_result():
    calc = InjuryRiskCalculator()
    result = calc.calculate_risk(
        fatigue_result=None,
        arm_slot_result=_arm_slot(0.0),
        timing_result=_timing(1.0),
        pitch_count=0,
        days_since_last_outing=4,
        hip_shoulder_separation_angle=30.0,
    )
    # Fatigue factor should be 0
    assert result["factor_values"]["fatigue_mechanics_drift"] == pytest.approx(0.0)


def test_none_arm_slot_result():
    calc = InjuryRiskCalculator()
    result = calc.calculate_risk(
        fatigue_result=_fatigue(0.0),
        arm_slot_result=None,
        timing_result=_timing(1.0),
        pitch_count=0,
        days_since_last_outing=4,
        hip_shoulder_separation_angle=30.0,
    )
    assert result["factor_values"]["arm_slot_instability"] == pytest.approx(0.0)


def test_none_timing_result():
    calc = InjuryRiskCalculator()
    result = calc.calculate_risk(
        fatigue_result=_fatigue(0.0),
        arm_slot_result=_arm_slot(0.0),
        timing_result=None,
        pitch_count=0,
        days_since_last_outing=4,
        hip_shoulder_separation_angle=30.0,
    )
    assert result["factor_values"]["kinetic_chain_disruption"] == pytest.approx(0.0)


def test_missing_fatigue_score_field():
    calc = InjuryRiskCalculator()
    # Dict without fatigue_score key
    result = calc.calculate_risk(
        fatigue_result={"some_other_field": 1.0},
        arm_slot_result=_arm_slot(0.0),
        timing_result=_timing(1.0),
        pitch_count=0,
        days_since_last_outing=4,
        hip_shoulder_separation_angle=30.0,
    )
    assert result["factor_values"]["fatigue_mechanics_drift"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 9: Report generation
# ---------------------------------------------------------------------------

def test_report_non_empty():
    calc = InjuryRiskCalculator()
    risk_result = calc.calculate_risk(**_max_inputs())
    report = calc.generate_report(risk_result)
    assert isinstance(report, str)
    assert len(report) > 0


def test_report_contains_top_3_drivers():
    calc = InjuryRiskCalculator()
    risk_result = calc.calculate_risk(**_max_inputs())
    report = calc.generate_report(risk_result)
    # Should mention numbered drivers
    assert "1." in report
    assert "2." in report
    assert "3." in report


def test_report_contains_score():
    calc = InjuryRiskCalculator()
    risk_result = calc.calculate_risk(**_max_inputs())
    report = calc.generate_report(risk_result)
    assert "100.0" in report or "100" in report


def test_report_green_traffic_light():
    calc = InjuryRiskCalculator()
    risk_result = calc.calculate_risk(**_zero_inputs())
    report = calc.generate_report(risk_result)
    assert "GREEN" in report or "green" in report.lower()


# ---------------------------------------------------------------------------
# Test 10: Compare to average
# ---------------------------------------------------------------------------

def test_compare_above_average():
    calc = InjuryRiskCalculator()
    result = calc.compare_to_average(50.0, positional_avg=35.0)
    assert result["above_average"] is True
    assert result["delta"] == pytest.approx(15.0, abs=0.1)


def test_compare_below_average():
    calc = InjuryRiskCalculator()
    result = calc.compare_to_average(20.0, positional_avg=35.0)
    assert result["above_average"] is False
    assert result["delta"] == pytest.approx(-15.0, abs=0.1)


def test_compare_default_sp_avg():
    calc = InjuryRiskCalculator()
    result = calc.compare_to_average(40.0, position="SP")
    # Default SP avg is 35.0
    assert result["positional_avg"] == pytest.approx(35.0)
    assert result["above_average"] is True


def test_compare_at_average():
    calc = InjuryRiskCalculator()
    result = calc.compare_to_average(35.0, positional_avg=35.0)
    assert result["delta"] == pytest.approx(0.0, abs=0.1)
    assert result["above_average"] is False
