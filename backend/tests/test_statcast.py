"""Tests for Statcast fetching, matching, and correlation engine.

All external dependencies (pybaseball) are mocked.
"""

from __future__ import annotations

import math
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from backend.app.data.statcast import StatcastFetcher, _KEEP_COLS
from backend.app.analysis.correlation import (
    CorrelationEngine,
    CorrelationResult,
    PITCH_TYPE_GROUPS,
    RegressionResult,
)
from backend.app.models.pitch import PitchMetadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_statcast_df(n: int = 20, seed: int = 42) -> pd.DataFrame:
    """Synthetic Statcast DataFrame."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "game_date": ["2023-06-01"] * n,
        "pitcher": [543037] * n,
        "inning": list(range(1, n + 1)),
        "pitch_number": list(range(1, n + 1)),
        "pitch_type": rng.choice(["FF", "SL", "CH", "CU"], size=n).tolist(),
        "release_speed": rng.uniform(85, 98, n).tolist(),
        "release_spin_rate": rng.uniform(2000, 2800, n).tolist(),
        "woba_value": rng.uniform(0.0, 2.0, n).tolist(),
        "estimated_woba_using_speedangle": rng.uniform(0.0, 1.5, n).tolist(),
        "launch_speed": rng.uniform(70, 105, n).tolist(),
        "launch_angle": rng.uniform(-30, 50, n).tolist(),
        "effective_speed": rng.uniform(84, 97, n).tolist(),
        "release_extension": rng.uniform(5.5, 7.5, n).tolist(),
    })


def make_metadata(
    pitcher_id: str = "543037",
    game_date: datetime = datetime(2023, 6, 1),
    inning: int = 3,
    pitch_number: int = 3,
    pitch_type: str = "FF",
) -> PitchMetadata:
    return PitchMetadata(
        pitcher_id=pitcher_id,
        game_date=game_date,
        inning=inning,
        pitch_number=pitch_number,
        pitch_type=pitch_type,
    )


def make_features_df(n: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "max_shoulder_er": rng.uniform(140, 185, n),
        "hip_shoulder_sep": rng.uniform(25, 65, n),
        "stride_length": rng.uniform(0.7, 1.1, n),
        "elbow_flexion_at_release": rng.uniform(70, 120, n),
        "pitch_type": rng.choice(["FF", "SL", "CH", "CU"], size=n).tolist(),
    })


def make_outcomes_df(n: int = 30, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "release_speed": rng.uniform(85, 98, n),
        "release_spin_rate": rng.uniform(2000, 2800, n),
        "woba_value": rng.uniform(0.0, 2.0, n),
        "pitch_type": rng.choice(["FF", "SL", "CH", "CU"], size=n).tolist(),
    })


# ---------------------------------------------------------------------------
# StatcastFetcher tests
# ---------------------------------------------------------------------------

class TestStatcastFetcher:
    def test_load_csv(self, tmp_path):
        df = make_statcast_df(10)
        csv_path = tmp_path / "statcast.csv"
        df.to_csv(csv_path, index=False)

        fetcher = StatcastFetcher()
        result = fetcher.load_csv(csv_path)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10
        assert "release_speed" in result.columns

    def test_load_csv_only_keeps_known_cols(self, tmp_path):
        df = make_statcast_df(5)
        df["unknown_column"] = 0
        csv_path = tmp_path / "statcast.csv"
        df.to_csv(csv_path, index=False)

        fetcher = StatcastFetcher()
        result = fetcher.load_csv(csv_path)
        assert "unknown_column" not in result.columns

    def test_match_pitch_found(self):
        df = make_statcast_df(10)
        fetcher = StatcastFetcher()
        meta = make_metadata(pitcher_id="543037", inning=3, pitch_number=3)
        row = fetcher.match_pitch(df, meta)
        assert row is not None
        assert int(row["inning"]) == 3
        assert int(row["pitch_number"]) == 3

    def test_match_pitch_not_found(self):
        df = make_statcast_df(10)
        fetcher = StatcastFetcher()
        meta = make_metadata(pitcher_id="543037", inning=99, pitch_number=99)
        row = fetcher.match_pitch(df, meta)
        assert row is None

    def test_match_pitch_empty_df(self):
        fetcher = StatcastFetcher()
        meta = make_metadata()
        row = fetcher.match_pitch(pd.DataFrame(), meta)
        assert row is None

    def test_match_pitch_non_numeric_pitcher_id(self):
        df = make_statcast_df(5)
        fetcher = StatcastFetcher()
        meta = make_metadata(pitcher_id="not_a_number")
        row = fetcher.match_pitch(df, meta)
        assert row is None

    def test_match_all_returns_dict(self):
        df = make_statcast_df(10)
        fetcher = StatcastFetcher()
        metas = [
            make_metadata(pitch_number=i, inning=i)
            for i in range(1, 6)
        ]
        results = fetcher.match_all(df, metas)
        assert len(results) == 5
        assert all(pid in results for pid in [m.pitch_id for m in metas])

    def test_fetch_pitcher_uses_cache(self, tmp_path):
        df = make_statcast_df(5)
        fetcher = StatcastFetcher(cache_dir=tmp_path)
        # Pre-populate cache
        cache_file = tmp_path / "statcast_543037_2023-06-01_2023-10-01.parquet"
        df.to_parquet(cache_file, index=False)

        result = fetcher.fetch_pitcher(543037, "2023-06-01", "2023-10-01")
        assert len(result) == 5

    def test_fetch_pitcher_pybaseball_mock(self, tmp_path):
        df = make_statcast_df(8)
        fetcher = StatcastFetcher(cache_dir=tmp_path)

        mock_pb = MagicMock()
        mock_pb.statcast_pitcher.return_value = df

        with patch.dict("sys.modules", {"pybaseball": mock_pb}):
            result = fetcher.fetch_pitcher(543037, "2023-06-01", "2023-10-01")

        assert len(result) == 8
        mock_pb.statcast_pitcher.assert_called_once_with("2023-06-01", "2023-10-01", 543037)

    def test_fetch_pitcher_pybaseball_not_installed(self, tmp_path):
        fetcher = StatcastFetcher(cache_dir=tmp_path)
        with patch.dict("sys.modules", {"pybaseball": None}):
            result = fetcher.fetch_pitcher(543037, "2023-06-01", "2023-10-01")
        assert result.empty

    def test_fetch_pitcher_pybaseball_exception(self, tmp_path):
        fetcher = StatcastFetcher(cache_dir=tmp_path)
        mock_pb = MagicMock()
        mock_pb.statcast_pitcher.side_effect = RuntimeError("API down")
        with patch.dict("sys.modules", {"pybaseball": mock_pb}):
            result = fetcher.fetch_pitcher(543037, "2023-06-01", "2023-10-01")
        assert result.empty


# ---------------------------------------------------------------------------
# CorrelationEngine tests
# ---------------------------------------------------------------------------

class TestCorrelationEngine:
    def test_correlate_known_relationship(self):
        engine = CorrelationEngine()
        x = np.linspace(0, 10, 50)
        y = 2 * x + 0.1 * np.random.default_rng(0).normal(size=50)
        result = engine.correlate(x, y, "feature", "outcome")
        assert result.pearson_r > 0.99
        assert result.pearson_p < 0.001
        assert result.n == 50

    def test_correlate_insufficient_samples(self):
        engine = CorrelationEngine()
        x = np.array([1.0, 2.0])
        y = np.array([1.0, 2.0])
        result = engine.correlate(x, y, "f", "o")
        assert result.n == 2
        assert math.isnan(result.pearson_r)

    def test_correlate_with_nans(self):
        engine = CorrelationEngine()
        x = np.array([1.0, 2.0, float("nan"), 4.0, 5.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = engine.correlate(x, y, "f", "o")
        assert result.n == 4  # NaN row excluded

    def test_correlate_all_returns_list(self):
        engine = CorrelationEngine(outcomes=["release_speed", "release_spin_rate"])
        feat = make_features_df(30)
        out = make_outcomes_df(30)
        results = engine.correlate_all(feat, out, feature_cols=["max_shoulder_er", "stride_length"])
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, CorrelationResult) for r in results)

    def test_correlate_all_per_pitch_type(self):
        engine = CorrelationEngine(outcomes=["release_speed"])
        feat = make_features_df(60)
        out = make_outcomes_df(60)
        # Align pitch types
        out["pitch_type"] = feat["pitch_type"].values
        results = engine.correlate_all(feat, out, feature_cols=["max_shoulder_er"])
        pitch_types = {r.pitch_type for r in results}
        assert "all" in pitch_types
        # At least one pitch-type specific result
        assert len(pitch_types) > 1

    def test_significant_property(self):
        r = CorrelationResult("f", "o", "all", 50, 0.9, 0.001, 0.85, 0.002)
        assert r.significant

        r2 = CorrelationResult("f", "o", "all", 50, 0.05, 0.8, 0.05, 0.7)
        assert not r2.significant

    def test_regress_all_returns_list(self):
        engine = CorrelationEngine(outcomes=["release_speed"])
        feat = make_features_df(40)
        out = make_outcomes_df(40)
        results = engine.regress_all(feat, out, feature_cols=["max_shoulder_er", "stride_length"])
        assert isinstance(results, list)

    def test_regression_result_ranked_features(self):
        r = RegressionResult(
            outcome_name="release_speed",
            pitch_type="all",
            model_type="ridge",
            n=40,
            r2_score=0.5,
            feature_names=["a", "b", "c"],
            coefficients=[0.1, 0.8, 0.3],
        )
        assert r.ranked_features[0] == "b"  # highest |coef|

    def test_scatter_data_shape(self):
        engine = CorrelationEngine(outcomes=["release_speed"])
        feat = make_features_df(30)
        out = make_outcomes_df(30)
        scatter = engine.scatter_data(feat, out, "max_shoulder_er", "release_speed")
        assert len(scatter.x) == len(scatter.y)
        assert len(scatter.reg_x) == 2
        assert len(scatter.reg_y) == 2

    def test_scatter_data_per_pitch_type(self):
        engine = CorrelationEngine(outcomes=["release_speed"])
        feat = make_features_df(60)
        out = make_outcomes_df(60)
        out["pitch_type"] = feat["pitch_type"].values
        scatter = engine.scatter_data(feat, out, "max_shoulder_er", "release_speed", pitch_type="FB")
        assert len(scatter.x) > 0
        assert scatter.pitch_type == "FB"

    def test_regress_insufficient_samples(self):
        engine = CorrelationEngine(outcomes=["release_speed"])
        feat = make_features_df(3)
        out = make_outcomes_df(3)
        result = engine.regress(feat, out["release_speed"], ["max_shoulder_er"], "release_speed")
        assert result is None

    def test_pitch_type_groups_coverage(self):
        all_types = set()
        for types in PITCH_TYPE_GROUPS.values():
            all_types.update(types)
        assert "FF" in all_types
        assert "SL" in all_types
        assert "CU" in all_types
        assert "CH" in all_types
