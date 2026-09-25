"""Correlation with an interval, and the site-level association used by Fig. 2.

The per-site analysis correlates the site's nodal-axis error with its median
harvested node count and, as a control, correlates the tumour-axis error with the
same quantity. Both are Pearson correlations with a two-tailed test, and the control
is what makes the first informative.

Ref: Methods Sec. 2.4 (r = 0.91 between nodal error and harvested nodes; r = 0.14 for
the tumour axis); Sec. 4.11 (Pearson correlation, two-tailed test).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .bootstrap import percentile_interval


def pearson(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Pearson correlation with its two-tailed p-value and a Fisher interval."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3 or y.size != x.size:
        return {"r": float("nan"), "p": float("nan"), "low": float("nan"), "high": float("nan"), "n": float(x.size)}
    result = stats.pearsonr(x, y)
    r = float(result.statistic)
    p = float(result.pvalue)
    if abs(r) >= 1.0:
        low = high = r
    else:
        z = np.arctanh(r)
        se = 1.0 / np.sqrt(x.size - 3.0)
        low, high = float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))
    return {"r": r, "p": p, "low": low, "high": high, "n": float(x.size)}


def spearman(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Spearman correlation with its two-tailed p-value."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3 or y.size != x.size:
        return {"rho": float("nan"), "p": float("nan"), "low": float("nan"), "high": float("nan"), "n": float(x.size)}
    result = stats.spearmanr(x, y)
    return {"rho": float(result.statistic), "p": float(result.pvalue), "n": float(x.size)}


def spearman_with_bootstrap(x: np.ndarray, y: np.ndarray, resamples: int = 2000, alpha: float = 0.05, seed: int = 0) -> dict[str, float]:
    """Spearman correlation with a percentile bootstrap interval.

    Used for the in vitro correlation between the peritumoral texture feature and
    lymphatic vessel density, where the reported interval is a bootstrap one.

    Ref: Methods Sec. 2.10 and Sec. 4.10.
    """
    point = spearman(x, y)
    rng = np.random.default_rng(seed)
    count = x.size
    replicates = np.empty(resamples, dtype=np.float64)
    for position in range(resamples):
        draw = rng.integers(0, count, size=count)
        value = stats.spearmanr(x[draw], y[draw]).statistic
        replicates[position] = float(value) if np.isfinite(value) else np.nan
    low, high = percentile_interval(replicates, alpha)
    point.update({"low": low, "high": high, "replicates": float(resamples)})
    return point


def scan_handling_test(x: np.ndarray, y: np.ndarray, alpha: float = 0.05) -> dict[str, float]:
    """Permutation test of a correlation, used where a normal-theory p is not reported.

    The null is the permutation of one variable against the other, so no
    distributional assumption is made about the error statistic.

    Ref: Methods Sec. 4.11 (two-tailed test for the correlation among sites).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3:
        return {"r": float("nan"), "p_permutation": float("nan")}
    observed = abs(float(stats.pearsonr(x, y).statistic))
    rng = np.random.default_rng(0)
    count = 0
    trials = 20000
    for _ in range(trials):
        permuted = rng.permutation(y)
        if abs(float(stats.pearsonr(x, permuted).statistic)) >= observed:
            count += 1
    p = (count + 1) / (trials + 1)
    return {"r": observed, "p_permutation": float(p), "trials": float(trials), "alpha": alpha}
