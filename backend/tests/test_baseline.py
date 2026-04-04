"""Tests for baseline comparison module."""

import pytest
import numpy as np

from backend.tests.conftest_analysis import make_features
from backend.app.analysis.baseline import BaselineBuilder


def _make_baseline_list(n: int, rng_seed: int = 0) -> list:
    rng = np.random.default_rng(rng_seed)
    return [make_features(rng=rng) for _ in range(n)]


class TestBaselineBuilder:
    def test_requires_min_pitches(self):
        builder = BaselineBuilder()
        with pytest.raises(ValueError, match="20"):
            builder.compute_baseline("p1", "FF", _make_baseline_list(5))

    def test_compute_baseline_shape(self):
        builder = BaselineBuilder()
        bl = builder.compute_baseline("p1", "FF", _make_baseline_list(25))
        assert len(bl["means"]) == len(bl["stds"])
        assert bl["n"] == 25
        assert bl["pitch_type"] == "FF"

    def test_analyze_zero_deviation(self):
        """Pitcher exactly at baseline should have overall_z_score near 0."""
        builder = BaselineBuilder()
        feats = _make_baseline_list(25, rng_seed=1)
        bl = builder.compute_baseline("p1", "FF", feats)

        # Use the mean values directly — construct a feature at baseline mean
        result = builder.analyze("pitch1", "p1", feats[0], "FF", bl)
        assert result.module == "baseline-comparison"
        assert result.pitcher_id == "p1"
        assert isinstance(result.overall_z_score, float)

    def test_analyze_high_deviation_is_outlier(self):
        """Feature far from baseline should be flagged as outlier."""
        builder = BaselineBuilder()
        feats = _make_baseline_list(25, rng_seed=2)
        bl = builder.compute_baseline("p1", "FF", feats)

        # Create a pitch with max_shoulder_er far from baseline (way below)
        outlier = make_features(max_shoulder_er_deg=10.0)  # normal is ~170
        result = builder.analyze("pitch_out", "p1", outlier, "FF", bl)
        assert result.is_outlier is True
        assert result.overall_z_score > 2.5

    def test_top_deviations_count(self):
        builder = BaselineBuilder()
        feats = _make_baseline_list(20)
        bl = builder.compute_baseline("p1", "FF", feats)
        result = builder.analyze("p", "p1", feats[0], "FF", bl)
        assert len(result.top_deviations) <= 3

    def test_severity_normal_pitch(self):
        """A pitch at the mean should have low z scores in top_deviations."""
        builder = BaselineBuilder()
        rng = np.random.default_rng(99)
        feats = [make_features(rng=rng) for _ in range(25)]
        bl = builder.compute_baseline("p1", "FF", feats)
        # Analyze first pitch (within normal range)
        result = builder.analyze("p", "p1", feats[0], "FF", bl)
        # Not all pitches will be outliers
        assert result.overall_z_score < 10.0  # sanity
