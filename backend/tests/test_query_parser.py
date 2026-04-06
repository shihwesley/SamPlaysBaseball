"""Tests for query parser and orchestrator."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.app.query.parser import AnalysisQuery, QueryParser, _parse_json_response


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


# ---------------------------------------------------------------------------
# QueryParser with mocked Anthropic
# ---------------------------------------------------------------------------

class TestQueryParser:
    def test_parse_calls_api(self):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps({
            "pitcher_name": "Ohtani",
            "pitch_types": ["FF"],
            "comparison_mode": "time",
            "inning_range_a": [1, 2],
            "inning_range_b": [5, 6],
            "game_date": None,
            "concern": "fatigue",
        }))]

        with patch("backend.app.query.parser.Anthropic") as MockCls:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            MockCls.return_value = mock_client

            parser = QueryParser(api_key="test")
            q = parser.parse("Compare Ohtani's 1st inning fastballs to 5th inning")

            assert q.pitcher_name == "Ohtani"
            assert q.comparison_mode == "time"
            mock_client.messages.create.assert_called_once()

    def test_parse_with_grounding(self):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps({
            "pitcher_name": "Ohtani",
            "pitch_types": ["FF", "FC"],
            "comparison_mode": "type",
        }))]

        with patch("backend.app.query.parser.Anthropic") as MockCls:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            MockCls.return_value = mock_client

            parser = QueryParser(api_key="test")
            grounding = [
                {"name": "Ohtani", "pitch_types": ["FF", "SL", "CU", "FC"], "games": ["2026-04-01"]},
            ]
            q = parser.parse("Show me his cutter vs fastball", available_pitchers=grounding)

            # Check grounding was included in the prompt
            call_args = mock_client.messages.create.call_args
            user_msg = call_args.kwargs["messages"][0]["content"]
            assert "Ohtani" in user_msg
            assert "FF" in user_msg
