"""Tests for fatigue tracking and changepoint detection."""

import numpy as np

from backend.tests.conftest_analysis import make_features
from backend.app.analysis.fatigue import FatigueTracker
from backend.app.analysis.changepoint import detect_changepoints


def _make_game(n_fresh: int = 20, n_fatigued: int = 10, drift: float = 0.0) -> list:
    rng = np.random.default_rng(7)
    fresh = [make_features(rng=rng) for _ in range(n_fresh)]
    tired = [
        make_features(
            max_shoulder_er_deg=160.0 - drift,
            hip_shoulder_sep_deg=35.0,
            stride_length_normalized=0.70,
            release_point=[0.5 + drift * 0.01, 1.75, 0.0],
            rng=rng,
        )
        for _ in range(n_fatigued)
    ]
    return fresh + tired


class TestFatigueTracker:
    def test_fresh_baseline_uses_early_pitches(self):
        tracker = FatigueTracker()
        game = _make_game()
        bl = tracker.compute_fresh_baseline(game, n_early=10)
        assert "max_shoulder_er_deg" in bl
        assert abs(bl["max_shoulder_er_deg"] - 170.0) < 5.0

    def test_fatigue_score_increases_after_drift(self):
        tracker = FatigueTracker()
        game = _make_game(drift=20.0)
        bl = tracker.compute_fresh_baseline(game, n_early=10)

        # First pitch (no fatigue)
        r_fresh = tracker.analyze("p_fresh", "p1", game[0], 1, bl, game[:5])
        # Last pitch (fatigued)
        r_tired = tracker.analyze("p_tired", "p1", game[-1], 30, bl, game)

        assert r_tired.fatigue_score > r_fresh.fatigue_score

    def test_fatigue_markers_present(self):
        tracker = FatigueTracker()
        game = _make_game(drift=30.0)
        bl = tracker.compute_fresh_baseline(game)
        result = tracker.analyze("p", "p1", game[-1], 30, bl, game)
        assert len(result.markers) == 6  # one per indicator
        assert all(hasattr(m, "pct_change") for m in result.markers)

    def test_is_fatigued_threshold(self):
        tracker = FatigueTracker()
        # Large drift should trigger fatigue flag
        game = _make_game(drift=50.0)
        bl = tracker.compute_fresh_baseline(game, n_early=5)
        result = tracker.analyze("p", "p1", game[-1], 30, bl, game)
        assert result.is_fatigued is True

    def test_rolling_stats_length(self):
        tracker = FatigueTracker()
        game = _make_game()
        stats = tracker.rolling_stats(game, window=5)
        assert len(stats) == len(game)
        assert "max_shoulder_er_deg" in stats[0]


class TestChangepoint:
    def test_no_changepoint_in_flat_signal(self):
        flat = [1.0] * 30
        cps = detect_changepoints(flat)
        assert cps == []

    def test_detects_step_change(self):
        # Clear step change at index 15
        signal = [1.0] * 15 + [5.0] * 15
        cps = detect_changepoints(signal)
        assert len(cps) > 0
        # Changepoint should be near index 15
        assert any(10 <= cp <= 20 for cp in cps)

    def test_short_series_returns_empty(self):
        assert detect_changepoints([1.0, 2.0]) == []
