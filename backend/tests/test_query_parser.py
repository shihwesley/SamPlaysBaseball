"""Tests for query parser and orchestrator."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.app.query.parser import (
    AnalysisQuery,
    AnthropicBackend,
    Gemma4Backend,
    ParserBackend,
    QueryParser,
    _parse_json_response,
)


# ---------------------------------------------------------------------------
# AnalysisQuery validation
# ---------------------------------------------------------------------------

class TestAnalysisQuery:
    def test_valid_time_query(self):
        q = AnalysisQuery(
            pitcher_name="Ohtani",
            pitch_types=["FF"],
            comparison_mode="time",
            inning_range_a=(1, 3),
            inning_range_b=(5, 7),
        )
        assert q.validate() == []

    def test_time_query_missing_ranges(self):
        q = AnalysisQuery(
            pitcher_name="Ohtani",
            pitch_types=["FF"],
            comparison_mode="time",
        )
        errors = q.validate()
        assert any("inning_range" in e for e in errors)

    def test_type_query_needs_two_types(self):
        q = AnalysisQuery(
            pitcher_name="Ohtani",
            pitch_types=["FF"],
            comparison_mode="type",
        )
        errors = q.validate()
        assert any("2 pitch types" in e for e in errors)

    def test_valid_type_query(self):
        q = AnalysisQuery(
            pitcher_name="Ohtani",
            pitch_types=["FF", "FC"],
            comparison_mode="type",
        )
        assert q.validate() == []

    def test_valid_baseline_query(self):
        q = AnalysisQuery(
            pitcher_name="Ohtani",
            pitch_types=["SL"],
            comparison_mode="baseline",
        )
        assert q.validate() == []

    def test_invalid_mode(self):
        q = AnalysisQuery(
            pitcher_name="Ohtani",
            pitch_types=["FF"],
            comparison_mode="invalid",
        )
        errors = q.validate()
        assert any("comparison_mode" in e for e in errors)

    def test_missing_pitcher(self):
        q = AnalysisQuery(
            pitcher_name="",
            pitch_types=["FF"],
            comparison_mode="baseline",
        )
        errors = q.validate()
        assert any("pitcher_name" in e for e in errors)


# ---------------------------------------------------------------------------
# JSON response parsing
# ---------------------------------------------------------------------------

class TestParseJsonResponse:
    def test_basic_json(self):
        raw = json.dumps({
            "pitcher_name": "Ohtani",
            "pitch_types": ["FF"],
            "comparison_mode": "time",
            "inning_range_a": [1, 3],
            "inning_range_b": [5, 7],
            "game_date": None,
            "concern": "velocity",
        })
        q = _parse_json_response(raw)
        assert q.pitcher_name == "Ohtani"
        assert q.pitch_types == ["FF"]
        assert q.comparison_mode == "time"
        assert q.inning_range_a == (1, 3)
        assert q.inning_range_b == (5, 7)
        assert q.concern == "velocity"

    def test_markdown_fenced_json(self):
        raw = '```json\n{"pitcher_name": "Ohtani", "pitch_types": ["SL", "CU"], "comparison_mode": "type"}\n```'
        q = _parse_json_response(raw)
        assert q.pitcher_name == "Ohtani"
        assert q.comparison_mode == "type"
        assert q.pitch_types == ["SL", "CU"]

    def test_null_ranges(self):
        raw = json.dumps({
            "pitcher_name": "Ohtani",
            "pitch_types": ["FF"],
            "comparison_mode": "baseline",
            "inning_range_a": None,
            "inning_range_b": None,
        })
        q = _parse_json_response(raw)
        assert q.inning_range_a is None
        assert q.inning_range_b is None

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_json_response("not json at all")

    def test_strips_thinking_tags(self):
        raw = '<think>Let me analyze this query...</think>\n' + json.dumps({
            "pitcher_name": "Ohtani",
            "pitch_types": ["FF"],
            "comparison_mode": "baseline",
        })
        q = _parse_json_response(raw)
        assert q.pitcher_name == "Ohtani"
        assert q.comparison_mode == "baseline"


# ---------------------------------------------------------------------------
# Mock backend for testing
# ---------------------------------------------------------------------------

class MockBackend(ParserBackend):
    name = "mock"

    def __init__(self, response: str):
        self._response = response
        self.last_system = None
        self.last_user = None

    def complete(self, system: str, user: str, max_tokens: int = 300) -> str:
        self.last_system = system
        self.last_user = user
        return self._response


# ---------------------------------------------------------------------------
# QueryParser with mock backend
# ---------------------------------------------------------------------------

class TestQueryParser:
    def test_parse_returns_query(self):
        backend = MockBackend(json.dumps({
            "pitcher_name": "Ohtani",
            "pitch_types": ["FF"],
            "comparison_mode": "time",
            "inning_range_a": [1, 2],
            "inning_range_b": [5, 6],
            "game_date": None,
            "concern": "fatigue",
        }))

        parser = QueryParser(backend)
        q = parser.parse("Compare Ohtani's 1st inning fastballs to 5th inning")

        assert q.pitcher_name == "Ohtani"
        assert q.comparison_mode == "time"
        assert q.concern == "fatigue"

    def test_grounding_appears_in_prompt(self):
        backend = MockBackend(json.dumps({
            "pitcher_name": "Ohtani",
            "pitch_types": ["FF", "FC"],
            "comparison_mode": "type",
        }))

        parser = QueryParser(backend)
        grounding = [
            {"name": "Ohtani", "pitch_types": ["FF", "SL", "CU", "FC"], "games": ["2026-04-01"]},
        ]
        parser.parse("Show me his cutter vs fastball", available_pitchers=grounding)

        assert "Ohtani" in backend.last_user
        assert "FF" in backend.last_user
        assert "2026-04-01" in backend.last_user

    def test_local_constructor(self):
        # Just test that .local() creates the right backend type
        # (don't actually load the model)
        with patch("backend.app.query.parser.Gemma4Backend") as MockGemma:
            mock_instance = MagicMock()
            MockGemma.return_value = mock_instance
            parser = QueryParser.local()
            assert parser.backend is mock_instance

    def test_cloud_constructor(self):
        with patch("backend.app.query.parser.AnthropicBackend") as MockAnth:
            mock_instance = MagicMock()
            MockAnth.return_value = mock_instance
            parser = QueryParser.cloud(api_key="test-key")
            MockAnth.assert_called_once_with(api_key="test-key", model="claude-haiku-4-5-20251001")

    def test_anthropic_backend_calls_api(self):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="test response")]

        with patch("anthropic.Anthropic") as MockCls:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            MockCls.return_value = mock_client

            backend = AnthropicBackend(api_key="test-key")
            result = backend.complete("system", "user")

            assert result == "test response"
            mock_client.messages.create.assert_called_once()
