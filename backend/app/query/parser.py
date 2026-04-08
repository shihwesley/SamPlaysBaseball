"""LLM-powered query parser for mechanics analysis.

Takes free-text analyst queries, returns structured AnalysisQuery.
Grounded with actual pitcher/pitch type inventory from PitchDB.
Supports local (Gemma 4 via mlx-vlm) and cloud (Anthropic) backends.

Usage:
    # Local (default — no API key needed)
    parser = QueryParser.local()
    query = parser.parse("Compare Ohtani's first inning fastballs to his 6th inning")

    # Cloud fallback
    parser = QueryParser.cloud(api_key="sk-...")
    query = parser.parse("Why did his slider command fall off?")
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

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
    "concern": "tipping" | "release_consistency" | "velocity" | "getting hit" | null
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


# ---------------------------------------------------------------------------
# Parser backend interface
# ---------------------------------------------------------------------------

class ParserBackend(ABC):
    """Abstract interface for the LLM that powers query parsing."""

    name: str = "base"

    @abstractmethod
    def complete(self, system: str, user: str, max_tokens: int = 300) -> str:
        """Send system + user prompt, return raw text response."""
        ...


class Gemma4Backend(ParserBackend):
    """Local Gemma 4 E4B via mlx-vlm. No API key needed."""

    name = "gemma4"

    def __init__(self, model_id: str = "google/gemma-4-e4b-it"):
        self.model_id = model_id
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        from mlx_vlm import load
        self._model, self._processor = load(self.model_id)

    def complete(self, system: str, user: str, max_tokens: int = 300) -> str:
        self._load()
        from mlx_vlm import generate as vlm_generate
        from mlx_vlm.prompt_utils import apply_chat_template

        prompt = f"{system}\n\n{user}"
        formatted = apply_chat_template(
            self._processor,
            self._model.config,
            prompt,
            num_images=0,
        )

        return vlm_generate(
            model=self._model,
            processor=self._processor,
            prompt=formatted,
            image=None,
            max_tokens=max_tokens,
        )


class AnthropicBackend(ParserBackend):
    """Anthropic Claude API. Requires API key."""

    name = "anthropic"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-haiku-4-5-20251001",
    ):
        from anthropic import Anthropic
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def complete(self, system: str, user: str, max_tokens: int = 300) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text.strip()


# ---------------------------------------------------------------------------
# QueryParser
# ---------------------------------------------------------------------------

class QueryParser:
    """Parses natural language queries into AnalysisQuery.

    Provider-agnostic: pass any ParserBackend.
    """

    def __init__(self, backend: ParserBackend) -> None:
        self.backend = backend

    @classmethod
    def local(cls, model_id: str = "google/gemma-4-e4b-it") -> QueryParser:
        """Create a parser using local Gemma 4 E4B (no API key)."""
        return cls(backend=Gemma4Backend(model_id=model_id))

    @classmethod
    def cloud(cls, api_key: str | None = None, model: str = "claude-haiku-4-5-20251001") -> QueryParser:
        """Create a parser using Anthropic Claude API."""
        return cls(backend=AnthropicBackend(api_key=api_key, model=model))

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

        raw = self.backend.complete(
            system=_PARSER_SYSTEM_PROMPT,
            user=user_msg,
            max_tokens=300,
        )

        return _parse_json_response(raw)


def _parse_json_response(raw: str) -> AnalysisQuery:
    """Parse LLM JSON output into AnalysisQuery."""
    # Strip thinking tags if present (Gemma4 with thinking enabled)
    if "<think>" in raw:
        think_end = raw.find("</think>")
        if think_end != -1:
            raw = raw[think_end + len("</think>"):].strip()

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
