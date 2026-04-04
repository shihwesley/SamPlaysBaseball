"""Baseline comparison module: z-score deviation from pitcher's own baseline."""

from __future__ import annotations

import numpy as np

from backend.app.models.analysis import BaselineComparisonResult, JointDeviation
from backend.app.pipeline.features import BiomechFeatures

_FEATURE_NAMES = [
    "max_shoulder_er_deg",
    "hip_shoulder_sep_deg",
    "stride_length_normalized",
    "release_x",
    "release_y",
    "release_z",
    "phase_fp_to_mer",
    "phase_mer_to_release",
]

_MIN_PITCHES = 20


def _extract_scalars(f: BiomechFeatures) -> list[float]:
    return [
        f.max_shoulder_er_deg,
        f.hip_shoulder_sep_deg,
        f.stride_length_normalized,
        float(f.release_point[0]),
        float(f.release_point[1]),
        float(f.release_point[2]),
        float(f.phases.mer - f.phases.foot_plant),
        float(f.phases.release - f.phases.mer),
    ]


class BaselineBuilder:
    """Compute and apply baseline statistics for a pitcher."""

    def compute_baseline(
        self,
        pitcher_id: str,
        pitch_type: str,
        features_list: list[BiomechFeatures],
    ) -> dict:
        """Compute mean + std baseline from a list of pitches.

        Requires >= 20 pitches.
        """
        if len(features_list) < _MIN_PITCHES:
            raise ValueError(
                f"Need >= {_MIN_PITCHES} pitches to compute baseline, got {len(features_list)}"
            )

        matrix = np.array([_extract_scalars(f) for f in features_list])
        means = matrix.mean(axis=0).tolist()
        stds = np.where(matrix.std(axis=0) < 1e-9, 1.0, matrix.std(axis=0)).tolist()

        return {
            "pitcher_id": pitcher_id,
            "pitch_type": pitch_type,
            "feature_names": _FEATURE_NAMES,
            "means": means,
            "stds": stds,
            "n": len(features_list),
        }

    def analyze(
        self,
        pitch_id: str,
        pitcher_id: str,
        features: BiomechFeatures,
        pitch_type: str,
        baseline: dict,
    ) -> BaselineComparisonResult:
        """Return z-score deviations vs the provided baseline."""
        scalars = _extract_scalars(features)
        means = baseline["means"]
        stds = baseline["stds"]
        feature_names = baseline.get("feature_names", _FEATURE_NAMES)

        z_scores = [(v - m) / s for v, m, s in zip(scalars, means, stds)]
        overall_z = float(np.mean(np.abs(z_scores)))

        # Top 3 deviations by |z|
        indexed = sorted(enumerate(z_scores), key=lambda x: abs(x[1]), reverse=True)
        top_devs = [
            JointDeviation(
                joint_index=idx,
                joint_name=feature_names[idx] if idx < len(feature_names) else None,
                deviation_mm=float(scalars[idx] - means[idx]),
                z_score=float(z_scores[idx]),
            )
            for idx, _ in indexed[:3]
        ]

        return BaselineComparisonResult(
            pitch_id=pitch_id,
            pitcher_id=pitcher_id,
            pitch_type=pitch_type,
            overall_z_score=overall_z,
            top_deviations=top_devs,
            is_outlier=overall_z > 2.5,
        )
