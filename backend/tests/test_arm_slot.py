"""Tests for arm slot drift module."""

import numpy as np

from backend.tests.conftest_analysis import make_features
from backend.app.analysis.arm_slot import ArmSlotAnalyzer


def _make_stable_game(n: int = 20, rng_seed: int = 0) -> list:
    rng = np.random.default_rng(rng_seed)
    return [make_features(release_point=[0.5, 1.8, 0.0], rng=rng) for _ in range(n)]


def _make_drifted_game(n_stable: int = 20, n_drifted: int = 10, drift_deg: float = 5.0) -> list:
    rng = np.random.default_rng(42)
    stable = [make_features(release_point=[0.5, 1.8, 0.0], rng=rng) for _ in range(n_stable)]
    # Shift release point to change arm slot angle
    # arctan2(y, x): increase y to increase angle
    import math
    # baseline angle ~ arctan2(1.8, 0.5) ~ 74.5 deg
    # to shift by drift_deg, shift y
    target_angle = math.atan2(1.8, 0.5) + math.radians(drift_deg)
    r = math.sqrt(0.5**2 + 1.8**2)
    new_y = r * math.sin(target_angle)
    new_x = r * math.cos(target_angle)
    drifted = [
        make_features(release_point=[new_x, new_y, 0.0], rng=rng)
        for _ in range(n_drifted)
    ]
    return stable + drifted


class TestArmSlotAnalyzer:
    def test_compute_arm_slot_returns_float(self):
        analyzer = ArmSlotAnalyzer()
        f = make_features()
        slot = analyzer.compute_arm_slot(f)
        assert isinstance(slot, float)

    def test_stable_game_no_drift(self):
        analyzer = ArmSlotAnalyzer()
        game = _make_stable_game(20)
        bl = analyzer.compute_baseline(game[:10])
        result = analyzer.analyze("p", "p1", game[5], 5, game[:5], bl)
        assert result.is_significant_drift is False
        assert abs(result.drift_degrees) < 3.0

    def test_drifted_pitches_flagged(self):
        analyzer = ArmSlotAnalyzer()
        game = _make_drifted_game(drift_deg=8.0)
        bl = analyzer.compute_baseline(game[:10])
        result = analyzer.analyze("p_drift", "p1", game[-1], 30, game, bl)
        assert result.is_significant_drift is True
        assert abs(result.drift_degrees) > 3.0

    def test_cumulative_drift_computed(self):
        analyzer = ArmSlotAnalyzer()
        game = _make_stable_game(10)
        bl = analyzer.compute_baseline(game)
        result = analyzer.analyze("p", "p1", game[-1], 10, game, bl)
        assert result.cumulative_drift_degrees is not None

    def test_bimodal_unimodal_data(self):
        analyzer = ArmSlotAnalyzer()
        # Unimodal: all slots near 74 deg
        slots = [74.0 + np.random.default_rng(i).normal(0, 0.5) for i in range(20)]
        assert analyzer.detect_bimodal(slots) is False

    def test_bimodal_two_clusters(self):
        analyzer = ArmSlotAnalyzer()
        # Clearly bimodal: two clusters far apart
        cluster1 = [60.0 + np.random.default_rng(i).normal(0, 0.5) for i in range(30)]
        cluster2 = [80.0 + np.random.default_rng(i + 100).normal(0, 0.5) for i in range(30)]
        result = analyzer.detect_bimodal(cluster1 + cluster2)
        assert result is True
