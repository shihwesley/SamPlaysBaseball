"""Tests for timing analysis and energy decomposition."""

import numpy as np

from backend.tests.conftest_analysis import make_features
from backend.app.analysis.timing import TimingAnalyzer
from backend.app.analysis.energy import decompose_energy


class TestTimingAnalyzer:
    def test_valid_sequence_no_issue(self):
        """Proximal-to-distal sequence should give no timing issue."""
        analyzer = TimingAnalyzer()
        f = make_features(
            pelvis_peak=6, trunk_peak=9, shoulder_peak=12,
            elbow_peak=15, wrist_peak=18,
        )
        result = analyzer.analyze("p", "p1", f, "FF")
        assert result.is_timing_issue is False
        assert result.timing_score == 1.0

    def test_invalid_sequence_flagged(self):
        """Out-of-order peaks should be a timing issue."""
        analyzer = TimingAnalyzer()
        f = make_features(
            pelvis_peak=14, trunk_peak=12, shoulder_peak=10,  # reversed
            elbow_peak=8, wrist_peak=6,
        )
        result = analyzer.analyze("p", "p1", f, "FF")
        assert result.is_timing_issue is True
        assert result.timing_score < 1.0

    def test_events_count(self):
        analyzer = TimingAnalyzer()
        f = make_features()
        result = analyzer.analyze("p", "p1", f, "FF")
        # 8 events: foot_plant, 5 segment peaks, mer, ball_release
        assert len(result.events) == 8

    def test_timing_score_bounded(self):
        analyzer = TimingAnalyzer()
        f = make_features(
            pelvis_peak=18, trunk_peak=14, shoulder_peak=10,
            elbow_peak=6, wrist_peak=6,
        )
        result = analyzer.analyze("p", "p1", f, "FF")
        assert 0.0 <= result.timing_score <= 1.0

    def test_module_name(self):
        analyzer = TimingAnalyzer()
        result = analyzer.analyze("p", "p1", make_features(), "FF")
        assert result.module == "timing-analysis"

    def test_gap_too_small_flagged(self):
        """Segments peaking on same frame (gap=0) should be a timing issue."""
        analyzer = TimingAnalyzer()
        f = make_features(
            pelvis_peak=6, trunk_peak=6, shoulder_peak=6,
            elbow_peak=6, wrist_peak=6,
        )
        result = analyzer.analyze("p", "p1", f, "FF")
        assert result.is_timing_issue is True


class TestEnergyDecomposition:
    def test_keys_present(self):
        series = np.linspace([0, 0, 0], [0.1, 0.5, 1.0], 10)
        result = decompose_energy(series)
        assert set(result.keys()) == {"forward", "lateral", "vertical", "total", "wasted_motion_index"}

    def test_forward_dominant_low_waste(self):
        """If all motion is forward (z), wasted_motion_index should be low."""
        series = np.column_stack([
            np.zeros(10),
            np.zeros(10),
            np.linspace(0, 5, 10),  # pure forward
        ])
        result = decompose_energy(series)
        assert result["forward"] > 0
        assert result["wasted_motion_index"] < 0.1

    def test_empty_series_returns_zeros(self):
        result = decompose_energy(np.zeros((1, 3)))
        assert result["total"] == 0.0
        assert result["wasted_motion_index"] == 0.0

    def test_wasted_motion_bounded(self):
        series = np.random.default_rng(5).uniform(0, 1, (20, 3))
        result = decompose_energy(series)
        assert 0.0 <= result["wasted_motion_index"] <= 1.0
