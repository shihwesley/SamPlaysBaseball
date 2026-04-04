"""Fatigue tracking: rolling window stats and changepoint detection."""

from __future__ import annotations

import numpy as np

from backend.app.models.analysis import FatigueMarker, FatigueTrackingResult
from backend.app.pipeline.features import BiomechFeatures

_INDICATORS = [
    "max_shoulder_er_deg",
    "hip_shoulder_sep_deg",
    "stride_length_normalized",
    "release_x",
    "release_y",
    "release_z",
]

_MIN_EARLY = 5


def _indicator_values(f: BiomechFeatures) -> dict[str, float]:
    return {
        "max_shoulder_er_deg": f.max_shoulder_er_deg,
        "hip_shoulder_sep_deg": f.hip_shoulder_sep_deg,
        "stride_length_normalized": f.stride_length_normalized,
        "release_x": float(f.release_point[0]),
        "release_y": float(f.release_point[1]),
        "release_z": float(f.release_point[2]),
    }


class FatigueTracker:
    """Track fatigue indicators across pitches within a game."""

    def compute_fresh_baseline(
        self,
        features_list: list[BiomechFeatures],
        n_early: int = 15,
    ) -> dict:
        """Mean of first n_early pitches per indicator."""
        n = max(_MIN_EARLY, min(n_early, len(features_list)))
        early = features_list[:n]
        result: dict[str, float] = {}
        for name in _INDICATORS:
            vals = [_indicator_values(f)[name] for f in early]
            result[name] = float(np.mean(vals))
        return result

    def rolling_stats(
        self,
        features_list: list[BiomechFeatures],
        window: int = 5,
    ) -> list[dict]:
        """Rolling mean per indicator over window pitches."""
        out = []
        for i in range(len(features_list)):
            start = max(0, i - window + 1)
            chunk = features_list[start : i + 1]
            entry: dict[str, float] = {}
            for name in _INDICATORS:
                vals = [_indicator_values(f)[name] for f in chunk]
                entry[name] = float(np.mean(vals))
            out.append(entry)
        return out

    def analyze(
        self,
        pitch_id: str,
        pitcher_id: str,
        features: BiomechFeatures,
        pitch_number: int,
        fresh_baseline: dict,
        recent_features: list[BiomechFeatures],
    ) -> FatigueTrackingResult:
        """Compute fatigue score for the current pitch."""
        current = _indicator_values(features)
        markers: list[FatigueMarker] = []
        pct_changes: list[float] = []

        for name in _INDICATORS:
            baseline_val = fresh_baseline.get(name, 0.0)
            current_val = current[name]
            if abs(baseline_val) < 1e-9:
                pct_change = 0.0
            else:
                pct_change = (current_val - baseline_val) / abs(baseline_val)
            markers.append(
                FatigueMarker(
                    metric_name=name,
                    value=current_val,
                    baseline_value=baseline_val,
                    pct_change=pct_change,
                )
            )
            pct_changes.append(abs(pct_change))

        mean_pct = float(np.mean(pct_changes))
        # Sigmoid-like scaling: 20% change = score 1.0
        fatigue_score = min(1.0, mean_pct / 0.2)

        return FatigueTrackingResult(
            pitch_id=pitch_id,
            pitcher_id=pitcher_id,
            pitch_number_in_game=pitch_number,
            markers=markers,
            fatigue_score=fatigue_score,
            is_fatigued=fatigue_score > 0.6,
        )
