"""Tests for delivery comparison module."""

import numpy as np
import pytest

from backend.app.analysis.compare_deliveries import (
    DeliveryComparison,
    FeatureDiff,
    compare_deliveries,
)


def _synthetic_pitch(n_frames: int = 60, seed: int = 0) -> np.ndarray:
    """Generate synthetic MHR70 keypoints with realistic-ish motion."""
    rng = np.random.RandomState(seed)
    base = rng.randn(70, 3).astype(np.float64) * 0.3
    joints = np.tile(base, (n_frames, 1, 1))
    # Add smooth motion to arm joints (shoulder=6, elbow=8, wrist=41)
    t = np.linspace(0, np.pi, n_frames)
    for j_idx in [6, 8, 41]:
        joints[:, j_idx, 0] += np.sin(t) * 0.3  # forward swing
        joints[:, j_idx, 1] -= np.sin(t * 2) * 0.1  # vertical
    # Add leg stride (ankle=13)
    joints[:, 13, 2] -= np.linspace(0, 0.5, n_frames)
    joints[:, 13, 1] -= np.abs(np.sin(t)) * 0.15  # foot drop
    return joints


class TestCompareDeliveries:
    def test_returns_comparison_object(self):
        a = _synthetic_pitch(60, seed=0)
        b = _synthetic_pitch(60, seed=1)
        result = compare_deliveries(a, b, fps=30.0)
        assert isinstance(result, DeliveryComparison)

    def test_diffs_are_populated(self):
        a = _synthetic_pitch(60, seed=0)
        b = _synthetic_pitch(60, seed=1)
        result = compare_deliveries(a, b, fps=30.0)
        assert len(result.diffs) > 0
        assert all(isinstance(d, FeatureDiff) for d in result.diffs)

    def test_identical_pitches_have_zero_diff(self):
        a = _synthetic_pitch(60, seed=0)
        result = compare_deliveries(a, a.copy(), fps=30.0)
        for d in result.diffs:
            assert d.abs_diff == pytest.approx(0.0, abs=1e-6), f"{d.name} has nonzero diff"

    def test_aligned_arrays_shape(self):
        a = _synthetic_pitch(60, seed=0)
        b = _synthetic_pitch(80, seed=1)  # different length
        result = compare_deliveries(a, b, fps=30.0)
        assert result.aligned_a.shape == (101, 70, 3)
        assert result.aligned_b.shape == (101, 70, 3)

    def test_frame_distances_shape(self):
        a = _synthetic_pitch(60, seed=0)
        b = _synthetic_pitch(60, seed=1)
        result = compare_deliveries(a, b, fps=30.0)
        assert result.frame_distances.shape == (101,)
        assert np.all(result.frame_distances >= 0)

    def test_summary_contains_labels(self):
        a = _synthetic_pitch(60, seed=0)
        b = _synthetic_pitch(60, seed=1)
        result = compare_deliveries(
            a, b, fps=30.0, labels=("4-seam #1", "4-seam #2")
        )
        assert "4-seam #1" in result.summary
        assert "4-seam #2" in result.summary

    def test_top_differences_count(self):
        a = _synthetic_pitch(60, seed=0)
        b = _synthetic_pitch(60, seed=1)
        result = compare_deliveries(a, b, fps=30.0)
        assert len(result.top_differences(3)) == 3
        assert len(result.top_differences(1)) == 1

    def test_different_frame_counts(self):
        a = _synthetic_pitch(40, seed=0)
        b = _synthetic_pitch(90, seed=1)
        result = compare_deliveries(a, b, fps=30.0)
        assert result.aligned_a.shape == result.aligned_b.shape

    def test_left_handed(self):
        a = _synthetic_pitch(60, seed=0)
        b = _synthetic_pitch(60, seed=1)
        result = compare_deliveries(a, b, fps=30.0, handedness="left")
        assert isinstance(result, DeliveryComparison)
