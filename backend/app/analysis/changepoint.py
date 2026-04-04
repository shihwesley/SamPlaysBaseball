"""Changepoint detection: CUSUM with optional ruptures backend."""

from __future__ import annotations

import numpy as np


def detect_changepoints(series: list[float], method: str = "cusum") -> list[int]:
    """Detect changepoints in a 1D series.

    Args:
        series: 1D signal values.
        method: "cusum" (default) or "ruptures" (falls back to cusum if unavailable).

    Returns:
        List of frame indices where significant changes occur.
    """
    if len(series) < 4:
        return []

    arr = np.array(series, dtype=np.float64)

    if method == "ruptures":
        try:
            import ruptures as rpt  # type: ignore

            algo = rpt.Binseg(model="rbf").fit(arr.reshape(-1, 1))
            result = algo.predict(n_bkps=1)
            # ruptures returns breakpoints excluding last index
            return [r for r in result if r < len(arr)]
        except Exception:
            pass

    # CUSUM fallback
    return _cusum(arr)


def _cusum(arr: np.ndarray, threshold_factor: float = 4.0) -> list[int]:
    """Simple CUSUM changepoint detection.

    Uses a sliding reference: compare first half vs second half to set threshold.
    """
    n = len(arr)
    mu = arr.mean()
    std = arr.std()
    if std < 1e-9:
        return []

    # Scale threshold to number of samples so it catches single abrupt shifts
    threshold = threshold_factor * std
    cusum_pos = 0.0
    cusum_neg = 0.0
    changepoints: list[int] = []
    k = 0.5 * std  # allowance (slack)

    for i, x in enumerate(arr):
        cusum_pos = max(0.0, cusum_pos + (x - mu) - k)
        cusum_neg = max(0.0, cusum_neg - (x - mu) - k)
        if cusum_pos > threshold or cusum_neg > threshold:
            changepoints.append(i)
            # Reset but update running mean to track local level
            mu = float(np.mean(arr[max(0, i - 2) : i + 1]))
            cusum_pos = 0.0
            cusum_neg = 0.0

    return changepoints
