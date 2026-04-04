"""ReportGenerator — assembles scouting reports from stored analysis results."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from backend.app.models.storage import StorageLayer
from backend.app.reports import templates


class ScoutingReport(BaseModel):
    pitcher_id: str
    pitcher_name: str | None = None
    report_type: Literal["pitcher", "outing", "pitch_type"] = "pitcher"
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    sections: dict[str, str] = Field(default_factory=dict)
    narrative: str = ""
    recommendations: list[str] = Field(default_factory=list)
    risk_level: str = "green"
    metadata: dict = Field(default_factory=dict)


class ReportGenerator:
    def __init__(
        self,
        storage: StorageLayer,
        llm=None,  # LLMReportGenerator | None
    ) -> None:
        self.storage = storage
        self.llm = llm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_pitcher_report(self, pitcher_id: str) -> ScoutingReport:
        """Full per-pitcher report across all pitch types."""
        baseline = self.storage.load_baseline(pitcher_id)
        pitcher_name = baseline.pitcher_name if baseline else None
        pitch_types = list(baseline.by_pitch_type.keys()) if baseline else []
        sample_counts = (
            {pt: bl.sample_count for pt, bl in baseline.by_pitch_type.items()}
            if baseline else {}
        )

        # Aggregate analysis results
        pitch_ids = self.storage.list_pitch_ids(pitcher_id)
        tipping = self._latest(pitch_ids, "tipping-detection")
        fatigue = self._latest(pitch_ids, "fatigue-tracking")
        command = self._latest(pitch_ids, "command-analysis")
        arm_slot = self._latest(pitch_ids, "arm-slot-drift")
        risk_data = self._latest(pitch_ids, "injury-risk")

        sections: dict[str, str] = {
            "pitcher_profile": templates.pitcher_profile_section(
                pitcher_id, pitcher_name, pitch_types, sample_counts
            ),
            "tipping": templates.tipping_section(
                tipping.get("tip_signals", []) if tipping else [],
                tipping.get("max_separation_score", 0.0) if tipping else 0.0,
                tipping.get("is_tipping", False) if tipping else False,
            ),
            "fatigue": templates.fatigue_section(
                fatigue.get("markers", []) if fatigue else [],
                fatigue.get("fatigue_score", 0.0) if fatigue else 0.0,
                fatigue.get("is_fatigued", False) if fatigue else False,
            ),
            "command": templates.command_section(
                command.get("command_score", 0.5) if command else 0.5,
                command if command else None,
            ),
            "arm_slot": templates.arm_slot_section(
                arm_slot.get("drift_degrees") if arm_slot else None,
                arm_slot.get("baseline_arm_slot_degrees") if arm_slot else None,
                arm_slot.get("is_significant_drift", False) if arm_slot else False,
            ),
        }

        risk_score = 0.0
        traffic_light = "green"
        top_factors: list[tuple[str, float]] = []
        if risk_data:
            risk_score = risk_data.get("risk_score", 0.0)
            traffic_light = risk_data.get("traffic_light", "green")
            factor_values = risk_data.get("factor_values", {})
            top_factors = sorted(factor_values.items(), key=lambda x: x[1], reverse=True)[:3]
        sections["injury_risk"] = templates.injury_risk_section(
            risk_score, traffic_light, top_factors
        )

        narrative = self._build_narrative(sections, pitcher_name or pitcher_id)
        recommendations = self._build_recommendations(risk_data, {
            "fatigue_score": fatigue.get("fatigue_score", 0) if fatigue else 0,
            "command_score": command.get("command_score", 0.5) if command else 0.5,
            "is_tipping": tipping.get("is_tipping", False) if tipping else False,
        })

        return ScoutingReport(
            pitcher_id=pitcher_id,
            pitcher_name=pitcher_name,
            report_type="pitcher",
            sections=sections,
            narrative=narrative,
            recommendations=recommendations,
            risk_level=traffic_light,
            metadata={
                "pitch_count": len(pitch_ids),
                "pitch_types": pitch_types,
            },
        )

    def generate_outing_report(self, pitcher_id: str, outing_date: str) -> ScoutingReport:
        """Per-outing report for a specific date (YYYY-MM-DD)."""
        with self.storage._conn() as conn:
            rows = conn.execute(
                "SELECT pitch_id FROM pitches WHERE pitcher_id = ? AND game_date LIKE ? ORDER BY pitch_number",
                (pitcher_id, f"{outing_date}%"),
            ).fetchall()
        pitch_ids = [r["pitch_id"] for r in rows]

        baseline = self.storage.load_baseline(pitcher_id)
        pitcher_name = baseline.pitcher_name if baseline else None

        fatigue_results = []
        arm_slot_results = []
        for pid in pitch_ids:
            fatigue_results.extend(self.storage.load_analysis(pid, module="fatigue-tracking"))
            arm_slot_results.extend(self.storage.load_analysis(pid, module="arm-slot-drift"))

        last_fatigue = fatigue_results[-1] if fatigue_results else None
        last_arm_slot = arm_slot_results[-1] if arm_slot_results else None

        sections: dict[str, str] = {
            "pitcher_profile": f"Outing: {pitcher_name or pitcher_id} on {outing_date}. {len(pitch_ids)} pitches tracked.",
            "fatigue": templates.fatigue_section(
                last_fatigue.get("markers", []) if last_fatigue else [],
                last_fatigue.get("fatigue_score", 0.0) if last_fatigue else 0.0,
                last_fatigue.get("is_fatigued", False) if last_fatigue else False,
            ),
            "arm_slot": templates.arm_slot_section(
                last_arm_slot.get("drift_degrees") if last_arm_slot else None,
                last_arm_slot.get("baseline_arm_slot_degrees") if last_arm_slot else None,
                last_arm_slot.get("is_significant_drift", False) if last_arm_slot else False,
            ),
        }

        narrative = self._build_narrative(sections, pitcher_name or pitcher_id)
        return ScoutingReport(
            pitcher_id=pitcher_id,
            pitcher_name=pitcher_name,
            report_type="outing",
            sections=sections,
            narrative=narrative,
            recommendations=[],
            risk_level="green",
            metadata={"outing_date": outing_date, "pitch_count": len(pitch_ids)},
        )

    def generate_pitch_type_report(self, pitcher_id: str, pitch_type: str) -> ScoutingReport:
        """Per-pitch-type breakdown with specific numbers."""
        baseline = self.storage.load_baseline(pitcher_id)
        pitcher_name = baseline.pitcher_name if baseline else None
        pt_baseline = baseline.get_baseline(pitch_type) if baseline else None

        with self.storage._conn() as conn:
            rows = conn.execute(
                "SELECT pitch_id FROM pitches WHERE pitcher_id = ? AND pitch_type = ?",
                (pitcher_id, pitch_type),
            ).fetchall()
        pitch_ids = [r["pitch_id"] for r in rows]

        command_results = []
        timing_results = []
        for pid in pitch_ids:
            command_results.extend(self.storage.load_analysis(pid, module="command-analysis"))
            timing_results.extend(self.storage.load_analysis(pid, module="timing-analysis"))

        last_command = command_results[-1] if command_results else None
        last_timing = timing_results[-1] if timing_results else None

        sections: dict[str, str] = {
            "pitcher_profile": templates.pitcher_profile_section(
                pitcher_id, pitcher_name, [pitch_type],
                {pitch_type: pt_baseline.sample_count if pt_baseline else len(pitch_ids)},
            ),
            "command": templates.command_section(
                last_command.get("command_score", 0.5) if last_command else 0.5,
                last_command,
            ),
            "timing": templates.timing_section(
                last_timing.get("events", []) if last_timing else [],
                last_timing.get("timing_score", 1.0) if last_timing else 1.0,
                last_timing.get("is_timing_issue", False) if last_timing else False,
            ),
        }

        narrative = self._build_narrative(sections, f"{pitcher_name or pitcher_id} — {pitch_type}")
        return ScoutingReport(
            pitcher_id=pitcher_id,
            pitcher_name=pitcher_name,
            report_type="pitch_type",
            sections=sections,
            narrative=narrative,
            recommendations=[],
            risk_level="green",
            metadata={"pitch_type": pitch_type, "pitch_count": len(pitch_ids)},
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _latest(self, pitch_ids: list[str], module: str) -> dict | None:
        """Return the most recent stored analysis result for a module (single query)."""
        if not pitch_ids:
            return None
        # Use pitcher_id from any pitch — all share the same pitcher
        # Resolve pitcher_id from the first pitch_id via storage
        # Fall back to per-pitch if needed; primary path uses load_analysis_by_pitcher
        pitcher_id = self._pitcher_id_from_pitch_ids(pitch_ids)
        if pitcher_id:
            results = self.storage.load_analysis_by_pitcher(pitcher_id, module=module)
            return results[-1] if results else None
        # Fallback: iterate (should not normally happen)
        for pid in reversed(pitch_ids):
            results = self.storage.load_analysis(pid, module=module)
            if results:
                return results[-1]
        return None

    def _pitcher_id_from_pitch_ids(self, pitch_ids: list[str]) -> str | None:
        """Retrieve pitcher_id for the given pitch list via one SQL query."""
        if not pitch_ids:
            return None
        with self.storage._conn() as conn:
            row = conn.execute(
                "SELECT pitcher_id FROM pitches WHERE pitch_id = ?",
                (pitch_ids[0],),
            ).fetchone()
        return row["pitcher_id"] if row else None

    def _build_narrative(self, sections: dict[str, str], pitcher_name: str) -> str:
        if self.llm is not None:
            try:
                return self.llm.generate_narrative(sections, pitcher_name)
            except Exception:
                pass
        # Template fallback: concatenate sections
        return "\n\n".join(sections.values())

    def _build_recommendations(
        self, risk_data: dict | None, analysis_summary: dict
    ) -> list[str]:
        if self.llm is not None and risk_data:
            try:
                factor_values = risk_data.get("factor_values", {})
                return self.llm.generate_recommendations(factor_values, analysis_summary)
            except Exception:
                pass
        # Template fallback
        recs = []
        if analysis_summary.get("is_tipping"):
            recs.append("Address pitch tipping — review pre-release mechanics with video.")
        fatigue = analysis_summary.get("fatigue_score", 0)
        if fatigue > 0.6:
            recs.append("Reduce workload — fatigue score above threshold.")
        command = analysis_summary.get("command_score", 1.0)
        if command < 0.5:
            recs.append("Work on release point consistency — command score below 0.50.")
        return recs
