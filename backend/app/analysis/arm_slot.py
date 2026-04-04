"""Arm slot drift detection: within-game and cross-outing trends."""

from __future__ import annotations

import numpy as np

from backend.app.models.analysis import ArmSlotDriftResult
from backend.app.pipeline.features import BiomechFeatures

_DRIFT_THRESHOLD = 3.0  # degrees


class ArmSlotAnalyzer:
    """Detect arm slot drift within a game and across outings."""

    def compute_arm_slot(self, features: BiomechFeatures) -> float:
        """Compute arm slot angle in degrees from release point vector.

        Uses arctan2(y, x) of the release point as a proxy for arm slot height.
        """
        rp = features.release_point
        return float(np.degrees(np.arctan2(float(rp[1]), float(rp[0]))))

    def compute_baseline(self, features_list: list[BiomechFeatures]) -> dict:
        """Mean + std of arm slot over the provided pitches."""
        if not features_list:
            return {"mean": 0.0, "std": 1.0, "n": 0}

        slots = [self.compute_arm_slot(f) for f in features_list]
        mean = float(np.mean(slots))
        std = float(np.std(slots))
        return {"mean": mean, "std": max(std, 1e-9), "n": len(features_list)}

    def analyze(
        self,
        pitch_id: str,
        pitcher_id: str,
        features: BiomechFeatures,
        pitch_number: int,
        game_features: list[BiomechFeatures],
        baseline: dict,
    ) -> ArmSlotDriftResult:
        """Compute arm slot and drift for this pitch."""
        arm_slot = self.compute_arm_slot(features)
        baseline_mean = baseline.get("mean", arm_slot)
        drift = arm_slot - baseline_mean

        # Rolling 5-pitch cumulative drift from early game average
        cumulative_drift: float | None = None
        if len(game_features) >= 3:
            early_n = max(3, min(5, len(game_features) // 3))
            early_slots = [self.compute_arm_slot(f) for f in game_features[:early_n]]
            early_mean = float(np.mean(early_slots))
            cumulative_drift = arm_slot - early_mean

        return ArmSlotDriftResult(
            pitch_id=pitch_id,
            pitcher_id=pitcher_id,
            arm_slot_degrees=arm_slot,
            baseline_arm_slot_degrees=baseline_mean,
            drift_degrees=drift,
            cumulative_drift_degrees=cumulative_drift,
            is_significant_drift=abs(drift) > _DRIFT_THRESHOLD,
        )

    def detect_bimodal(self, arm_slots: list[float]) -> bool:
        """Return True if arm slot distribution is bimodal (GMM BIC test)."""
        if len(arm_slots) < 6:
            return False
        try:
            from sklearn.mixture import GaussianMixture

            arr = np.array(arm_slots).reshape(-1, 1)
            gm1 = GaussianMixture(n_components=1, random_state=42).fit(arr)
            gm2 = GaussianMixture(n_components=2, random_state=42).fit(arr)
            return bool(gm2.bic(arr) < gm1.bic(arr) - 10.0)
        except Exception:
            return False
