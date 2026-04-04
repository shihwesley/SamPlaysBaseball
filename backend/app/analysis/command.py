"""Command analysis: release point consistency and mechanical predictors."""

from __future__ import annotations

import numpy as np

from backend.app.models.analysis import CommandAnalysisResult
from backend.app.pipeline.features import BiomechFeatures

_THRESHOLD_MM = 50.0  # 50mm = 5cm release scatter threshold


class CommandAnalyzer:
    """Analyze release point consistency and command quality."""

    def compute_release_stats(
        self,
        features_list: list[BiomechFeatures],
    ) -> dict:
        """3D centroid + std of release_point across all pitches."""
        if not features_list:
            return {"centroid": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0], "n": 0}

        points = np.array([f.release_point for f in features_list])  # (N, 3)
        centroid = points.mean(axis=0).tolist()
        std = np.where(points.std(axis=0) < 1e-9, 1.0, points.std(axis=0)).tolist()

        return {"centroid": centroid, "std": std, "n": len(features_list)}

    def analyze(
        self,
        pitch_id: str,
        pitcher_id: str,
        features: BiomechFeatures,
        pitch_type: str,
        release_stats: dict,
        plate_x: float | None = None,
        plate_z: float | None = None,
    ) -> CommandAnalysisResult:
        """Score command for this pitch."""
        centroid = release_stats["centroid"]
        rp = features.release_point

        # Deviations in mm (positions are in meters -> *1000)
        dx_mm = float((rp[0] - centroid[0]) * 1000.0)
        dz_mm = float((rp[2] - centroid[2]) * 1000.0)

        scatter_mm = float(np.sqrt(dx_mm**2 + dz_mm**2))
        command_score = max(0.0, 1.0 - scatter_mm / _THRESHOLD_MM)

        return CommandAnalysisResult(
            pitch_id=pitch_id,
            pitcher_id=pitcher_id,
            pitch_type=pitch_type,
            plate_x=plate_x,
            plate_z=plate_z,
            release_x_deviation=dx_mm,
            release_z_deviation=dz_mm,
            command_score=command_score,
        )
