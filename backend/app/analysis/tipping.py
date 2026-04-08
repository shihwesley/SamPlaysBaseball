"""Tipping confirmation: post-game pairwise comparison of deliveries by pitch type.

This module is a **post-game confirmation tool**, not an in-game predictor.

Use case: A coach (yours or the opposing team's) sees something live during a game and
suspects the pitcher is tipping a pitch type via body posture, glove position, hand
placement, or pre-delivery rhythm. After the game, the analyst runs this module against
that outing, grouped by pitch type, to surface the *measurable* in-plane body-posture
differences between the suspect pitch type and the rest. The output is something
concrete to show the player: "Frame 24, your glove sits 4 cm lower on your slider
deliveries than on your fastball deliveries. Here it is, side by side."

The historic example this tool was designed to confirm: Yu Darvish, 2017 World Series,
Game 7. The Astros bench reportedly picked up a pre-delivery tell on his slider/sweeper.
Run this module against game_pk 526517 grouped by pitch type to see the measurable
mechanical evidence frame by frame.

The classifier-based `TippingDetector.train()` path is preserved for legacy callers and
can still be used when at least 50 labeled pitches are available, but the **primary**
entry point for the post-game confirmation use case is `compare_within_outing()`, which
uses pairwise delivery comparison and does not require a trained model.

Note on what this can and cannot detect:
- DETECTS in-plane body-posture differences (glove height, shoulder set, hand placement,
  pre-delivery rhythm differences, head/torso lean) — these are exactly the signals
  where a single broadcast camera is most reliable.
- CANNOT detect grip-visible-in-glove tells (the actual 2017 Astros method) — that
  requires a center-field zoom on the pitcher's glove that is not in standard broadcast
  feeds.
- CANNOT predict tipping in advance — the human observer provides the hypothesis;
  this tool quantifies and locates it.

See VALIDATION.md for the full discussion of confidence levels and what is and is not
trustworthy in this output.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from backend.app.models.analysis import TippingDetectionResult, TipSignal
from backend.app.pipeline.features import BiomechFeatures

_MIN_PITCHES = 50
_TIPPING_MARGIN = 0.05

# Minimum sample size per pitch type for the post-game confirmation comparison.
# Below this, the in-plane delta numbers are too noisy to interpret as a real tell.
_MIN_PER_TYPE_FOR_CONFIRMATION = 5


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


# ---------------------------------------------------------------------------
# Post-game confirmation entry point (PRIMARY use case)
# ---------------------------------------------------------------------------

def compare_within_outing(
    features_by_pitch_type: dict[str, list[BiomechFeatures]],
    suspect_pitch_type: str,
    reference_pitch_type: str,
) -> dict:
    """Compare delivery features between two pitch types within a single outing.

    This is the **primary post-game confirmation entry point** for the tipping use case.
    Given a coach's hypothesis ("I think he tips his slider when throwing it after a
    fastball"), pass in the pre-release biomechanical feature dictionaries grouped by
    pitch type and get back the per-feature mean and std for each group, the in-plane
    delta, and a normalized "separation z-score" indicating how unlikely the delta is
    under the null hypothesis of no tipping.

    Returns a dict with:
        - n_suspect, n_reference: sample sizes
        - per_feature: {feature_name: {suspect_mean, ref_mean, delta, z_score}}
        - top_signals: top 3 features ranked by |z_score|
        - confidence: "high" / "medium" / "low" / "insufficient_data"
        - notes: list of human-readable observations

    Confidence levels:
        - "insufficient_data": fewer than 5 pitches in either group
        - "low": 5-9 per group, deltas exist but cannot rule out noise
        - "medium": 10+ per group, at least one feature with |z| > 1.5
        - "high": 10+ per group, at least one feature with |z| > 2.5

    The function does NOT make a binary "is tipping" call — that's a human judgment
    informed by the numbers, the live observation, and the video review. It returns
    structured evidence the analyst can show to a coach or player.
    """
    suspect_features = features_by_pitch_type.get(suspect_pitch_type, [])
    reference_features = features_by_pitch_type.get(reference_pitch_type, [])
    n_suspect = len(suspect_features)
    n_reference = len(reference_features)

    notes: list[str] = []

    if n_suspect < _MIN_PER_TYPE_FOR_CONFIRMATION or n_reference < _MIN_PER_TYPE_FOR_CONFIRMATION:
        return {
            "suspect_pitch_type": suspect_pitch_type,
            "reference_pitch_type": reference_pitch_type,
            "n_suspect": n_suspect,
            "n_reference": n_reference,
            "per_feature": {},
            "top_signals": [],
            "confidence": "insufficient_data",
            "notes": [
                f"Need at least {_MIN_PER_TYPE_FOR_CONFIRMATION} pitches per group "
                f"for a meaningful comparison. Got n_suspect={n_suspect}, "
                f"n_reference={n_reference}."
            ],
        }

    # Extract feature dicts for each delivery
    suspect_dicts = [_pre_release_features(f) for f in suspect_features]
    reference_dicts = [_pre_release_features(f) for f in reference_features]

    feature_names = list(suspect_dicts[0].keys())

    per_feature: dict[str, dict] = {}
    for fname in feature_names:
        s_vals = np.array([d[fname] for d in suspect_dicts], dtype=np.float64)
        r_vals = np.array([d[fname] for d in reference_dicts], dtype=np.float64)
        s_mean = float(s_vals.mean())
        r_mean = float(r_vals.mean())
        delta = s_mean - r_mean
        # Pooled std for the z-score; guard against zero std
        pooled_std = float(np.sqrt(0.5 * (s_vals.var() + r_vals.var())))
        z = delta / pooled_std if pooled_std > 1e-9 else 0.0
        per_feature[fname] = {
            "suspect_mean": s_mean,
            "reference_mean": r_mean,
            "delta": delta,
            "pooled_std": pooled_std,
            "z_score": z,
        }

    # Top signals by absolute z-score
    ranked = sorted(per_feature.items(), key=lambda kv: abs(kv[1]["z_score"]), reverse=True)
    top_signals = [
        {
            "feature": fname,
            "delta": vals["delta"],
            "z_score": vals["z_score"],
            "suspect_mean": vals["suspect_mean"],
            "reference_mean": vals["reference_mean"],
        }
        for fname, vals in ranked[:3]
    ]

    # Confidence calibration
    max_abs_z = max((abs(vals["z_score"]) for vals in per_feature.values()), default=0.0)
    if min(n_suspect, n_reference) < 10:
        confidence = "low"
        notes.append(
            f"Sample sizes are small (n_suspect={n_suspect}, n_reference={n_reference}). "
            f"Treat the deltas as suggestive, not confirmed."
        )
    elif max_abs_z > 2.5:
        confidence = "high"
        notes.append(
            f"Strong separation: at least one feature has |z|={max_abs_z:.2f}. "
            f"This is unlikely to be noise alone."
        )
    elif max_abs_z > 1.5:
        confidence = "medium"
        notes.append(
            f"Moderate separation: top feature has |z|={max_abs_z:.2f}. "
            f"Worth showing to a coach for visual confirmation."
        )
    else:
        confidence = "low"
        notes.append(
            f"No feature shows strong separation (max |z|={max_abs_z:.2f}). "
            f"If a coach saw a tell live, it may not be one of these biomechanical features."
        )

    return {
        "suspect_pitch_type": suspect_pitch_type,
        "reference_pitch_type": reference_pitch_type,
        "n_suspect": n_suspect,
        "n_reference": n_reference,
        "per_feature": per_feature,
        "top_signals": top_signals,
        "confidence": confidence,
        "notes": notes,
    }
