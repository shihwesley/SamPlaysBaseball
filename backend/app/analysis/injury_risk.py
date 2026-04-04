"""Composite injury risk indicator.

Aggregates fatigue, arm slot, and timing analysis into a 0-100 risk score
based on ASMI biomechanics research weightings.

This is a risk *indicator*, not a *predictor*. No injury outcome data is used.
Scores reflect published biomechanical load correlates only.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Risk factor weights (from ASMI research)
# ---------------------------------------------------------------------------

_WEIGHTS = {
    "fatigue_mechanics_drift": 0.30,
    "kinetic_chain_disruption": 0.25,
    "arm_slot_instability": 0.20,
    "workload": 0.15,
    "hip_shoulder_separation_deficit": 0.10,
}

# Positional average risk scores (SP = starting pitcher baseline estimate)
_DEFAULT_POSITIONAL_AVG = {
    "SP": 35.0,
    "RP": 30.0,
    "CP": 28.0,
}

# Arm slot drift normalization ceiling: 10 degrees = max instability
_ARM_SLOT_DRIFT_MAX = 10.0

# Traffic light bands
_TRAFFIC_LIGHT = [
    (0, 30, "green"),
    (31, 60, "yellow"),
    (61, 80, "orange"),
    (81, 100, "red"),
]

# Recommendations keyed by factor name
_RECOMMENDATIONS = {
    "fatigue_mechanics_drift": (
        "Mechanical drift detected. Consider reducing pitch count or increasing "
        "rest. Monitor fastball velocity and hip-shoulder separation on next outing."
    ),
    "kinetic_chain_disruption": (
        "Kinetic chain sequencing is out of order. Work with pitching coach on "
        "lower-half drive and late trunk rotation to reduce elbow stress."
    ),
    "arm_slot_instability": (
        "Arm slot is inconsistent. Drill flat-ground work with video feedback "
        "to reinforce repeatable release pattern."
    ),
    "workload": (
        "High workload load. Apply standard pitch-count protocols and enforce "
        "minimum 4-day rest before next appearance."
    ),
    "hip_shoulder_separation_deficit": (
        "Hip-shoulder separation below 30 degrees. Address hip mobility and "
        "rotational sequencing drills to reduce compensatory arm acceleration."
    ),
}


@dataclass
class InjuryRiskCalculator:
    """Calculate composite injury risk from upstream analysis results."""

    # Per-pitcher trend history: {pitcher_id: [score, score, ...]}
    _trend_history: dict[str, list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )

    # ---------------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------------

    def calculate_risk(
        self,
        fatigue_result,
        arm_slot_result,
        timing_result,
        pitch_count: int,
        days_since_last_outing: int,
        hip_shoulder_separation_angle: float,
    ) -> dict:
        """Return full risk report dict.

        Parameters
        ----------
        fatigue_result:
            FatigueTrackingResult or None.
        arm_slot_result:
            ArmSlotDriftResult or None.
        timing_result:
            TimingAnalysisResult or None.
        pitch_count:
            Total pitches thrown this outing.
        days_since_last_outing:
            Days since previous appearance (0 = back-to-back).
        hip_shoulder_separation_angle:
            Hip-shoulder separation in degrees (typical 20-60).
        """
        factors = self._extract_factors(
            fatigue_result,
            arm_slot_result,
            timing_result,
            pitch_count,
            days_since_last_outing,
            hip_shoulder_separation_angle,
        )

        # Weighted sum -> 0-100 score
        raw = sum(_WEIGHTS[k] * factors[k] for k in _WEIGHTS)
        score = min(100.0, max(0.0, round(raw * 100, 1)))

        # Per-factor contribution as % of current risk
        contributions: dict[str, float] = {}
        if score > 0:
            for k in _WEIGHTS:
                contrib = _WEIGHTS[k] * factors[k] * 100
                contributions[k] = round(contrib / score * 100, 1)
        else:
            contributions = {k: 0.0 for k in _WEIGHTS}

        return {
            "risk_score": score,
            "traffic_light": self._traffic_light(score),
            "factor_values": {k: round(factors[k], 4) for k in factors},
            "factor_contributions_pct": contributions,
            "weights": dict(_WEIGHTS),
        }

    def update_trend(self, pitcher_id: str, risk_score: float) -> dict:
        """Append risk_score to pitcher history and return trend info."""
        history = self._trend_history[pitcher_id]
        history.append(risk_score)

        # Detect 3+ consecutive increases
        is_increasing = False
        streak = 0
        if len(history) >= 2:
            streak = 1
            for i in range(len(history) - 1, 0, -1):
                if history[i] > history[i - 1]:
                    streak += 1
                else:
                    break
            is_increasing = streak >= 3

        return {
            "pitcher_id": pitcher_id,
            "history": list(history),
            "latest_score": risk_score,
            "increasing_trend": is_increasing,
            "consecutive_increases": streak if len(history) >= 2 else 0,
        }

    def compare_to_average(
        self,
        risk_score: float,
        positional_avg: Optional[float] = None,
        position: str = "SP",
    ) -> dict:
        """Compare pitcher risk to positional average."""
        avg = positional_avg if positional_avg is not None else _DEFAULT_POSITIONAL_AVG.get(position, 33.0)
        delta = risk_score - avg
        return {
            "risk_score": risk_score,
            "positional_avg": avg,
            "position": position,
            "delta": round(delta, 1),
            "above_average": delta > 0,
            "pct_above_avg": round(delta / avg * 100, 1) if avg > 0 else 0.0,
        }

    def generate_report(self, risk_result: dict) -> str:
        """Return a human-readable risk summary with top 3 drivers."""
        score = risk_result["risk_score"]
        light = risk_result["traffic_light"]
        contribs = risk_result["factor_contributions_pct"]
        factors = risk_result["factor_values"]

        # Top 3 contributors by contribution percentage
        ranked = sorted(contribs.items(), key=lambda x: x[1], reverse=True)[:3]

        lines = [
            f"Injury Risk Score: {score:.1f}/100 ({light.upper()})",
            "",
            "Top risk drivers:",
        ]
        for i, (factor, pct) in enumerate(ranked, 1):
            label = factor.replace("_", " ").title()
            raw_val = factors.get(factor, 0.0)
            lines.append(f"  {i}. {label}: {raw_val:.2f} normalized ({pct:.1f}% of current risk)")

        lines.append("")
        lines.append("Recommended adjustments:")
        for factor, _ in ranked:
            if factors.get(factor, 0) > 0:
                lines.append(f"  - {_RECOMMENDATIONS[factor]}")

        if score <= 30:
            lines.append("")
            lines.append("Overall: Risk is within normal range. Standard monitoring.")
        elif score <= 60:
            lines.append("")
            lines.append("Overall: Elevated risk. Monitor closely and consider workload management.")
        elif score <= 80:
            lines.append("")
            lines.append("Overall: High risk. Strong consideration for reduced workload or rest.")
        else:
            lines.append("")
            lines.append("Overall: Immediate concern. Recommend removing from current appearance.")

        return "\n".join(lines)

    # ---------------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------------

    def _extract_factors(
        self,
        fatigue_result,
        arm_slot_result,
        timing_result,
        pitch_count: int,
        days_since_last_outing: int,
        hip_shoulder_separation_angle: float,
    ) -> dict[str, float]:
        return {
            "fatigue_mechanics_drift": self._fatigue_factor(fatigue_result),
            "kinetic_chain_disruption": self._kinetic_chain_factor(timing_result),
            "arm_slot_instability": self._arm_slot_factor(arm_slot_result),
            "workload": self._workload_factor(pitch_count, days_since_last_outing),
            "hip_shoulder_separation_deficit": self._hip_shoulder_factor(
                hip_shoulder_separation_angle
            ),
        }

    def _fatigue_factor(self, fatigue_result) -> float:
        """Extract fatigue score (already 0-1) from FatigueTrackingResult."""
        if fatigue_result is None:
            return 0.0
        # Support both object attribute and dict access
        score = _get(fatigue_result, "fatigue_score", None)
        if score is None:
            return 0.0
        return float(min(1.0, max(0.0, score)))

    def _arm_slot_factor(self, arm_slot_result) -> float:
        """Normalize drift_degrees against _ARM_SLOT_DRIFT_MAX ceiling."""
        if arm_slot_result is None:
            return 0.0
        drift = _get(arm_slot_result, "drift_degrees", None)
        if drift is None:
            # Fall back to cumulative drift if available
            drift = _get(arm_slot_result, "cumulative_drift_degrees", None)
        if drift is None:
            return 0.0
        return float(min(1.0, abs(drift) / _ARM_SLOT_DRIFT_MAX))

    def _kinetic_chain_factor(self, timing_result) -> float:
        """Convert timing_score (1=perfect, 0=broken) to disruption (0-1)."""
        if timing_result is None:
            return 0.0
        timing_score = _get(timing_result, "timing_score", None)
        if timing_score is None:
            return 0.0
        return float(min(1.0, max(0.0, 1.0 - timing_score)))

    def _workload_factor(self, pitch_count: int, days_since_last_outing: int) -> float:
        """Combined workload: 60% pitch count + 40% rest deficit."""
        normalized_pitches = min(1.0, pitch_count / 100.0)
        # 0 days rest = 1.0 deficit, 4+ days = 0.0
        rest_deficit = max(0.0, 1.0 - days_since_last_outing / 4.0)
        return round(0.6 * normalized_pitches + 0.4 * rest_deficit, 4)

    def _hip_shoulder_factor(self, angle: float) -> float:
        """Below 30 degrees = deficit; 0 degrees = full deficit (1.0)."""
        return float(max(0.0, (30.0 - angle) / 30.0))

    @staticmethod
    def _traffic_light(score: float) -> str:
        for lo, hi, label in _TRAFFIC_LIGHT:
            if lo <= score <= hi:
                return label
        return "red"


def _get(obj, attr: str, default):
    """Attribute or dict key access with fallback."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)
