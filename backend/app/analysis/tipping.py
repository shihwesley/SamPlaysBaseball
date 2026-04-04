"""Tipping detection: identify mechanical tells that leak pitch type pre-release."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from backend.app.models.analysis import TippingDetectionResult, TipSignal
from backend.app.pipeline.features import BiomechFeatures

_MIN_PITCHES = 50
_TIPPING_MARGIN = 0.05


def _pre_release_features(f: BiomechFeatures) -> dict[str, float]:
    """Scalar summaries of kinematics up to MER (pre-release window)."""
    end = f.phases.mer + 1
    windows = {
        "elbow_flex": f.elbow_flexion[:end],
        "shoulder_er": f.shoulder_er[:end],
        "trunk_rot": f.trunk_rotation[:end],
        "trunk_tilt": f.trunk_tilt[:end],
    }
    result: dict[str, float] = {}
    for name, arr in windows.items():
        if len(arr) == 0:
            result[f"{name}_mean"] = 0.0
            result[f"{name}_std"] = 0.0
            result[f"{name}_max"] = 0.0
        else:
            result[f"{name}_mean"] = float(arr.mean())
            result[f"{name}_std"] = float(arr.std())
            result[f"{name}_max"] = float(arr.max())
    result["hip_shoulder_sep"] = f.hip_shoulder_sep_deg
    result["stride_length"] = f.stride_length_normalized
    return result


def _features_to_array(
    features_list: list[BiomechFeatures],
    feature_names: list[str],
) -> NDArray:
    rows = []
    for f in features_list:
        d = _pre_release_features(f)
        rows.append([d[k] for k in feature_names])
    return np.array(rows, dtype=np.float64)


class TippingDetector:
    """Detect pitch-type tipping from pre-release mechanics."""

    def train(
        self,
        pitcher_id: str,
        features_list: list[BiomechFeatures],
        pitch_types: list[str],
    ) -> dict:
        """Train classifier on pre-release features. Returns model dict."""
        if len(features_list) < _MIN_PITCHES:
            raise ValueError(
                f"Need >= {_MIN_PITCHES} labeled pitches to train tipping model, "
                f"got {len(features_list)}"
            )

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        y = le.fit_transform(pitch_types)
        unique_types = list(le.classes_)
        chance_level = 1.0 / len(unique_types)

        sample_feats = _pre_release_features(features_list[0])
        feature_names = list(sample_feats.keys())

        X = _features_to_array(features_list, feature_names)

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        cv_scores = cross_val_score(clf, X, y, cv=min(5, len(unique_types)))
        cv_accuracy = float(cv_scores.mean())

        clf.fit(X, y)

        from backend.app.analysis.shap_utils import shap_feature_importance
        importances = shap_feature_importance(clf, X, feature_names)

        is_tipping = cv_accuracy > chance_level + _TIPPING_MARGIN

        return {
            "pitcher_id": pitcher_id,
            "model": clf,
            "label_encoder": le,
            "feature_names": feature_names,
            "unique_types": unique_types,
            "cv_accuracy": cv_accuracy,
            "chance_level": chance_level,
            "is_tipping": is_tipping,
            "feature_importances": importances,
        }

    def analyze(
        self,
        pitch_id: str,
        pitcher_id: str,
        features: BiomechFeatures,
        pitch_type: str,
        trained_model: dict,
    ) -> TippingDetectionResult:
        """Evaluate tipping risk for a single pitch."""
        feature_names: list[str] = trained_model["feature_names"]
        importances: dict[str, float] = trained_model["feature_importances"]
        unique_types: list[str] = trained_model["unique_types"]
        is_tipping: bool = trained_model["is_tipping"]

        feat_dict = _pre_release_features(features)

        # Build tip signals for top-3 important features across each pair
        signals: list[TipSignal] = []
        sorted_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:3]
        for feat_name, importance in sorted_feats:
            if len(unique_types) >= 2:
                signals.append(
                    TipSignal(
                        feature_name=feat_name,
                        pitch_type_a=unique_types[0],
                        pitch_type_b=unique_types[1] if len(unique_types) > 1 else unique_types[0],
                        separation_score=min(1.0, importance),
                        frame_range=(0, features.phases.mer),
                    )
                )

        max_sep = max((s.separation_score for s in signals), default=0.0)

        return TippingDetectionResult(
            pitch_id=pitch_id,
            pitcher_id=pitcher_id,
            tip_signals=signals,
            max_separation_score=max_sep,
            is_tipping=is_tipping,
        )
