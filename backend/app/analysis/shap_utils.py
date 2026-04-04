"""SHAP feature importance with graceful fallback."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def shap_feature_importance(
    model,
    X: NDArray,
    feature_names: list[str],
) -> dict[str, float]:
    """Compute mean absolute SHAP values. Falls back to feature_importances_ if shap unavailable."""
    try:
        import shap  # type: ignore

        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)
        # For multi-class, shap_vals is a list; take mean over classes
        if isinstance(shap_vals, list):
            shap_arr = np.mean([np.abs(sv) for sv in shap_vals], axis=0)
        else:
            shap_arr = np.abs(shap_vals)
        mean_abs = shap_arr.mean(axis=0)
    except Exception:
        # Fallback to built-in importances
        if hasattr(model, "feature_importances_"):
            mean_abs = model.feature_importances_
        else:
            mean_abs = np.ones(len(feature_names)) / len(feature_names)

    return {name: float(val) for name, val in zip(feature_names, mean_abs)}
