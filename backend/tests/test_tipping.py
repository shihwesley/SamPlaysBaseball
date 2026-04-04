"""Tests for tipping detection module."""

import numpy as np
import pytest

from backend.tests.conftest_analysis import make_features
from backend.app.analysis.tipping import TippingDetector


def _make_labeled_pitches(
    n_per_type: int = 40,
    inject_difference: bool = False,
) -> tuple[list, list[str]]:
    """Create pitches with 2 pitch types. inject_difference=True makes them separable."""
    rng = np.random.default_rng(42)
    feats = []
    labels = []

    for i in range(n_per_type):
        feats.append(make_features(
            max_shoulder_er_deg=170.0 + (10.0 if inject_difference else 0.0),
            rng=rng,
        ))
        labels.append("FF")

    for i in range(n_per_type):
        feats.append(make_features(
            max_shoulder_er_deg=140.0 if inject_difference else 170.0,
            rng=rng,
        ))
        labels.append("CH")

    return feats, labels


class TestTippingDetector:
    def test_requires_min_pitches(self):
        detector = TippingDetector()
        feats, labels = _make_labeled_pitches(n_per_type=10)
        with pytest.raises(ValueError, match="50"):
            detector.train("p1", feats, labels)

    def test_tipping_detected_with_large_difference(self):
        detector = TippingDetector()
        feats, labels = _make_labeled_pitches(n_per_type=40, inject_difference=True)
        model = detector.train("p1", feats, labels)
        assert model["is_tipping"] is True
        assert model["cv_accuracy"] > model["chance_level"]

    def test_not_tipping_with_identical_features(self):
        detector = TippingDetector()
        feats, labels = _make_labeled_pitches(n_per_type=40, inject_difference=False)
        model = detector.train("p1", feats, labels)
        # With identical features, accuracy should be near chance
        assert model["cv_accuracy"] <= model["chance_level"] + 0.15

    def test_analyze_returns_result(self):
        detector = TippingDetector()
        feats, labels = _make_labeled_pitches(n_per_type=40, inject_difference=True)
        model = detector.train("p1", feats, labels)
        result = detector.analyze("pitch1", "p1", feats[0], "FF", model)
        assert result.module == "tipping-detection"
        assert result.pitcher_id == "p1"
        assert isinstance(result.max_separation_score, float)
        assert len(result.tip_signals) > 0

    def test_tip_signal_frame_range(self):
        detector = TippingDetector()
        feats, labels = _make_labeled_pitches(n_per_type=40, inject_difference=True)
        model = detector.train("p1", feats, labels)
        result = detector.analyze("p", "p1", feats[0], "FF", model)
        for sig in result.tip_signals:
            if sig.frame_range is not None:
                assert sig.frame_range[1] == feats[0].phases.mer
