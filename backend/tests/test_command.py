"""Tests for command analysis module."""

import numpy as np

from backend.tests.conftest_analysis import make_features
from backend.app.analysis.command import CommandAnalyzer


def _make_cluster(n: int = 100, spread_m: float = 0.002, rng_seed: int = 0) -> list:
    """Create pitches with tight release point cluster (spread in meters)."""
    rng = np.random.default_rng(rng_seed)
    return [
        make_features(
            release_point=[
                0.5 + rng.normal(0, spread_m),
                1.8,
                rng.normal(0, spread_m),
            ],
            rng=rng,
        )
        for _ in range(n)
    ]


def _make_outliers(n: int = 5, offset_m: float = 0.1) -> list:
    rng = np.random.default_rng(99)
    return [
        make_features(
            release_point=[0.5 + offset_m, 1.8, offset_m],
            rng=rng,
        )
        for _ in range(n)
    ]


class TestCommandAnalyzer:
    def test_tight_cluster_high_score(self):
        analyzer = CommandAnalyzer()
        feats = _make_cluster(100, spread_m=0.002)
        stats = analyzer.compute_release_stats(feats)
        result = analyzer.analyze("p", "p1", feats[0], "FF", stats)
        assert result.command_score > 0.8

    def test_outlier_low_score(self):
        analyzer = CommandAnalyzer()
        feats = _make_cluster(100, spread_m=0.002)
        stats = analyzer.compute_release_stats(feats)
        outliers = _make_outliers(1, offset_m=0.2)  # 200mm from centroid
        result = analyzer.analyze("p_out", "p1", outliers[0], "FF", stats)
        assert result.command_score < 0.5

    def test_release_stats_centroid_shape(self):
        analyzer = CommandAnalyzer()
        feats = _make_cluster(20)
        stats = analyzer.compute_release_stats(feats)
        assert len(stats["centroid"]) == 3
        assert len(stats["std"]) == 3
        assert stats["n"] == 20

    def test_empty_features_list(self):
        analyzer = CommandAnalyzer()
        stats = analyzer.compute_release_stats([])
        assert stats["n"] == 0

    def test_plate_location_passthrough(self):
        analyzer = CommandAnalyzer()
        feats = _make_cluster(20)
        stats = analyzer.compute_release_stats(feats)
        result = analyzer.analyze("p", "p1", feats[0], "FF", stats, plate_x=0.1, plate_z=2.5)
        assert result.plate_x == 0.1
        assert result.plate_z == 2.5

    def test_deviations_in_mm(self):
        analyzer = CommandAnalyzer()
        feats = _make_cluster(20, spread_m=0.001)
        stats = analyzer.compute_release_stats(feats)
        result = analyzer.analyze("p", "p1", feats[0], "FF", stats)
        # Deviations should be in mm range (small for tight cluster)
        assert result.release_x_deviation is not None
        assert abs(result.release_x_deviation) < 100.0
