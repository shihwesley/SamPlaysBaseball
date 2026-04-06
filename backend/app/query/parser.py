"""LLM-powered query parser for mechanics analysis.

Takes free-text analyst queries, returns structured AnalysisQuery.
Grounded with actual pitcher/pitch type inventory from PitchDB.

Usage:
    parser = QueryParser(api_key="...")
    query = parser.parse("Compare Ohtani's first inning fastballs to his 6th inning")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from anthropic import Anthropic

logger = logging.getLogger(__name__)


@dataclass
class AnalysisQuery:
    """Structured query extracted from natural language."""

    pitcher_name: str
    pitch_types: list[str]           # e.g. ["FF"] or ["FF", "FC"]
    comparison_mode: str             # "time" | "type" | "baseline"
    inning_range_a: tuple[int, int] | None = None
    inning_range_b: tuple[int, int] | None = None
    game_date: str | None = None     # "2026-04-01" or None (most recent)
    concern: str | None = None       # "command" | "velocity" | "getting hit"

    def validate(self) -> list[str]:
        """Return list of validation errors (empty = valid)."""
        errors = []
        if not self.pitcher_name:
            errors.append("pitcher_name is required")
        if self.comparison_mode not in ("time", "type", "baseline"):
            errors.append(f"invalid comparison_mode: {self.comparison_mode!r}")
        if self.comparison_mode == "time" and not (self.inning_range_a and self.inning_range_b):
            errors.append("time comparison requires inning_range_a and inning_range_b")
        if self.comparison_mode == "type" and len(self.pitch_types) < 2:
            errors.append("type comparison requires at least 2 pitch types")
        return errors


_PARSER_SYSTEM_PROMPT = """\
You are a baseball query parser. Extract structured analysis parameters from \
natural language queries about pitcher mechanics.

Output ONLY valid JSON matching this schema:
{
    "pitcher_name": "last name or full name",
    "pitch_types": ["FF", "SL", ...],
    "comparison_mode": "time" | "type" | "baseline",
    "inning_range_a": [start, end] or null,
    "inning_range_b": [start, end] or null,
    "game_date": "YYYY-MM-DD" or null,
    "concern": "command" | "velocity" | "getting hit" | "fatigue" | "tipping" | null
}

Rules:
- comparison_mode "time": same pitch type, different innings. Requires inning ranges.
- comparison_mode "type": different pitch types, same game. Requires 2+ pitch types.
- comparison_mode "baseline": current vs career average. Single pitch type.
- If no innings specified, use null for ranges.
- If no game date, use null (system uses most recent).
- Pitch type codes: FF=4-seam, SI=sinker, FC=cutter, SL=slider, CU=curveball, \
CH=changeup, FS=splitter, KC=knuckle-curve, KN=knuckleball, ST=sweeper, SV=slurve.\
"""


class QueryParser:
    """Parses natural language queries into AnalysisQuery."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-haiku-4-5-20251001",
    ) -> None:
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def parse(
        self,
        text: str,
        available_pitchers: list[dict] | None = None,
    ) -> AnalysisQuery:
        """Parse free-text query into structured AnalysisQuery.

        Args:
            text: Natural language query from the analyst.
            available_pitchers: Grounding context — list of dicts with
                "name", "pitch_types", "games" fields. Prevents hallucination.
        """
        grounding = ""
        if available_pitchers:
            lines = []
            for p in available_pitchers:
                types = ", ".join(p.get("pitch_types", []))
                games = ", ".join(p.get("games", []))
                lines.append(f"  {p['name']} (types: {types}; games: {games})")
            grounding = "\nAvailable pitchers:\n" + "\n".join(lines) + "\n"

        user_msg = f"{grounding}\nQuery: {text}"

        response = self.client.messages.create(
            model=self.model,
            max_tokens=300,
            system=_PARSER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = response.content[0].text.strip()

        return _parse_json_response(raw)


def _parse_json_response(raw: str) -> AnalysisQuery:
    """Parse LLM JSON output into AnalysisQuery."""
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

    data = json.loads(raw)

    inning_a = tuple(data["inning_range_a"]) if data.get("inning_range_a") else None
    inning_b = tuple(data["inning_range_b"]) if data.get("inning_range_b") else None

    return AnalysisQuery(
        pitcher_name=data["pitcher_name"],
        pitch_types=data.get("pitch_types", []),
        comparison_mode=data["comparison_mode"],
        inning_range_a=inning_a,
        inning_range_b=inning_b,
        game_date=data.get("game_date"),
        concern=data.get("concern"),
    )
