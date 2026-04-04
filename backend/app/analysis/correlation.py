"""Correlation and regression engine: biomechanical features vs Statcast outcomes."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Outcome columns from Statcast that we can correlate against.
DEFAULT_OUTCOMES = [
    "release_speed",
    "release_spin_rate",
    "woba_value",
    "estimated_woba_using_speedangle",
    "launch_speed",
    "launch_angle",
    "effective_speed",
    "release_extension",
]

# Pitch type groupings (coarse)
PITCH_TYPE_GROUPS = {
    "FB": ["FF", "SI", "FC"],
    "SL": ["SL"],
    "CB": ["CU", "KC"],
    "CH": ["CH", "FS"],
}


@dataclass
class CorrelationResult:
    """Per-feature correlation with one outcome metric."""

    feature_name: str
    outcome_name: str
    pitch_type: str  # "all" or FB/SL/CB/CH
    n: int
    pearson_r: float
    pearson_p: float
    spearman_r: float
    spearman_p: float

    @property
    def significant(self) -> bool:
        return self.pearson_p < 0.05 or self.spearman_p < 0.05


@dataclass
class RegressionResult:
    """Ridge or LASSO regression: feature importances for one outcome."""

    outcome_name: str
    pitch_type: str
    model_type: str  # "ridge" or "lasso"
    n: int
    r2_score: float
    feature_names: list[str]
    coefficients: list[float]
    ranked_features: list[str] = field(default_factory=list)  # sorted by |coef|

    def __post_init__(self) -> None:
        if not self.ranked_features and self.feature_names:
            pairs = sorted(
                zip(self.feature_names, self.coefficients),
                key=lambda x: abs(x[1]),
                reverse=True,
            )
            self.ranked_features = [p[0] for p in pairs]


@dataclass
class ScatterData:
    """Data for a scatter plot with regression line."""

    feature_name: str
    outcome_name: str
    pitch_type: str
    x: list[float]
    y: list[float]
    reg_x: list[float]
    reg_y: list[float]


class CorrelationEngine:
    """Correlate biomechanical features with Statcast outcomes.

    Usage:
        engine = CorrelationEngine()
        # features_df: rows=pitches, cols=feature values + 'pitch_type'
        # statcast_df: rows=pitches, cols=Statcast outcomes + 'pitch_type'
        correlations = engine.correlate_all(features_df, statcast_df)
        regressions = engine.regress_all(features_df, statcast_df)
    """

    def __init__(
        self,
        outcomes: list[str] | None = None,
        ridge_alpha: float = 1.0,
        lasso_alpha: float = 0.01,
    ) -> None:
        self.outcomes = outcomes or DEFAULT_OUTCOMES
        self.ridge_alpha = ridge_alpha
        self.lasso_alpha = lasso_alpha

    # ------------------------------------------------------------------
    # Correlation
    # ------------------------------------------------------------------

    def correlate(
        self,
        x: np.ndarray,
        y: np.ndarray,
        feature_name: str,
        outcome_name: str,
        pitch_type: str = "all",
    ) -> CorrelationResult:
        """Pearson + Spearman correlation between one feature and one outcome."""
        mask = np.isfinite(x) & np.isfinite(y)
        x_clean = x[mask]
        y_clean = y[mask]
        n = len(x_clean)

        if n < 3:
            return CorrelationResult(
                feature_name=feature_name,
                outcome_name=outcome_name,
                pitch_type=pitch_type,
                n=n,
                pearson_r=float("nan"),
                pearson_p=float("nan"),
                spearman_r=float("nan"),
                spearman_p=float("nan"),
            )

        pr, pp = stats.pearsonr(x_clean, y_clean)
        sr, sp = stats.spearmanr(x_clean, y_clean)

        return CorrelationResult(
            feature_name=feature_name,
            outcome_name=outcome_name,
            pitch_type=pitch_type,
            n=n,
            pearson_r=float(pr),
            pearson_p=float(pp),
            spearman_r=float(sr),
            spearman_p=float(sp),
        )

    def correlate_all(
        self,
        features_df: pd.DataFrame,
        outcomes_df: pd.DataFrame,
        feature_cols: list[str] | None = None,
    ) -> list[CorrelationResult]:
        """Run correlation for all feature × outcome pairs, overall + per pitch type.

        Args:
            features_df: DataFrame with biomech feature columns + optional 'pitch_type'.
            outcomes_df: DataFrame with Statcast outcome columns + optional 'pitch_type'.
            feature_cols: subset of columns to use (defaults to all numeric non-pitch_type).

        Returns:
            List of CorrelationResult objects.
        """
        if feature_cols is None:
            feature_cols = [
                c for c in features_df.select_dtypes(include=np.number).columns
                if c != "pitch_type"
            ]

        outcome_cols = [c for c in self.outcomes if c in outcomes_df.columns]
        results: list[CorrelationResult] = []

        # Align on index
        idx = features_df.index.intersection(outcomes_df.index)
        feat = features_df.loc[idx]
        out = outcomes_df.loc[idx]

        # Overall
        for fc in feature_cols:
            for oc in outcome_cols:
                r = self.correlate(
                    feat[fc].to_numpy(dtype=float),
                    out[oc].to_numpy(dtype=float),
                    feature_name=fc,
                    outcome_name=oc,
                    pitch_type="all",
                )
                results.append(r)

        # Per pitch type
        if "pitch_type" in feat.columns:
            for group_name, pitch_types in PITCH_TYPE_GROUPS.items():
                mask = feat["pitch_type"].isin(pitch_types)
                if mask.sum() < 3:
                    continue
                f_sub = feat[mask]
                o_sub = out[mask]
                for fc in feature_cols:
                    for oc in outcome_cols:
                        r = self.correlate(
                            f_sub[fc].to_numpy(dtype=float),
                            o_sub[oc].to_numpy(dtype=float),
                            feature_name=fc,
                            outcome_name=oc,
                            pitch_type=group_name,
                        )
                        results.append(r)

        return results

    # ------------------------------------------------------------------
    # Regression
    # ------------------------------------------------------------------

    def regress(
        self,
        features_df: pd.DataFrame,
        outcome_series: pd.Series,
        feature_cols: list[str],
        outcome_name: str,
        pitch_type: str = "all",
        model_type: str = "ridge",
    ) -> RegressionResult | None:
        """Fit Ridge or LASSO regression for one outcome.

        Returns None if fewer than 5 samples after cleaning.
        """
        try:
            from sklearn.linear_model import Lasso, Ridge
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import r2_score as sk_r2
        except ImportError:
            logger.error("scikit-learn not installed")
            return None

        idx = features_df.index.intersection(outcome_series.index)
        X = features_df.loc[idx, feature_cols].to_numpy(dtype=float)
        y = outcome_series.loc[idx].to_numpy(dtype=float)

        # Drop rows with any NaN
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X, y = X[mask], y[mask]

        if len(y) < 5:
            return None

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if model_type == "ridge":
            model = Ridge(alpha=self.ridge_alpha)
        else:
            model = Lasso(alpha=self.lasso_alpha, max_iter=5000)

        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)
        r2 = float(sk_r2(y, y_pred))

        return RegressionResult(
            outcome_name=outcome_name,
            pitch_type=pitch_type,
            model_type=model_type,
            n=len(y),
            r2_score=r2,
            feature_names=feature_cols,
            coefficients=[float(c) for c in model.coef_],
        )

    def regress_all(
        self,
        features_df: pd.DataFrame,
        outcomes_df: pd.DataFrame,
        feature_cols: list[str] | None = None,
        model_type: str = "ridge",
    ) -> list[RegressionResult]:
        """Fit regressions for all outcomes, overall + per pitch type."""
        if feature_cols is None:
            feature_cols = [
                c for c in features_df.select_dtypes(include=np.number).columns
                if c != "pitch_type"
            ]

        outcome_cols = [c for c in self.outcomes if c in outcomes_df.columns]
        idx = features_df.index.intersection(outcomes_df.index)
        feat = features_df.loc[idx]
        out = outcomes_df.loc[idx]

        results: list[RegressionResult] = []

        for oc in outcome_cols:
            r = self.regress(feat, out[oc], feature_cols, oc, "all", model_type)
            if r:
                results.append(r)

        if "pitch_type" in feat.columns:
            for group_name, pitch_types in PITCH_TYPE_GROUPS.items():
                mask = feat["pitch_type"].isin(pitch_types)
                if mask.sum() < 5:
                    continue
                for oc in outcome_cols:
                    r = self.regress(
                        feat[mask], out.loc[mask, oc],
                        feature_cols, oc, group_name, model_type,
                    )
                    if r:
                        results.append(r)

        return results

    # ------------------------------------------------------------------
    # Visualization data
    # ------------------------------------------------------------------

    def scatter_data(
        self,
        features_df: pd.DataFrame,
        outcomes_df: pd.DataFrame,
        feature_name: str,
        outcome_name: str,
        pitch_type: str = "all",
    ) -> ScatterData:
        """Generate x/y points and regression line for a scatter plot.

        Args:
            features_df: biomech features DataFrame.
            outcomes_df: Statcast outcomes DataFrame.
            feature_name: column in features_df.
            outcome_name: column in outcomes_df.
            pitch_type: "all" or group name.

        Returns:
            ScatterData with raw points and regression line.
        """
        idx = features_df.index.intersection(outcomes_df.index)
        feat = features_df.loc[idx]
        out = outcomes_df.loc[idx]

        if pitch_type != "all" and "pitch_type" in feat.columns:
            pt_list = PITCH_TYPE_GROUPS.get(pitch_type, [pitch_type])
            mask = feat["pitch_type"].isin(pt_list)
            feat = feat[mask]
            out = out[mask]

        x = feat[feature_name].to_numpy(dtype=float)
        y = out[outcome_name].to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]

        reg_x: list[float] = []
        reg_y: list[float] = []

        if len(x) >= 2:
            slope, intercept, *_ = stats.linregress(x, y)
            x_min, x_max = float(x.min()), float(x.max())
            reg_x = [x_min, x_max]
            reg_y = [slope * x_min + intercept, slope * x_max + intercept]

        return ScatterData(
            feature_name=feature_name,
            outcome_name=outcome_name,
            pitch_type=pitch_type,
            x=x.tolist(),
            y=y.tolist(),
            reg_x=reg_x,
            reg_y=reg_y,
        )
