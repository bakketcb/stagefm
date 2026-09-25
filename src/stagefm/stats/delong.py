"""DeLong variance and interval for the area under the ROC curve.

The interval is the DeLong interval the manuscript reports on every axis: the
placement-value decomposition gives the AUC variance without resampling, and the
interval is the normal approximation on the logit-free scale.

The implementation is written against the definitions rather than by calling a
library, because the same placement values are reused for the macro and ordinal
summaries and because the test suite checks it against brute-force pairwise
counting.

Ref: Methods Sec. 4.11 (discrimination summarised with DeLong intervals).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class DeLongResult:
    """AUC, its standard error, and the two-sided interval."""

    auc: float
    variance: float
    standard_error: float
    low: float
    high: float
    n_positive: int
    n_negative: int


def midrank(values: np.ndarray) -> np.ndarray:
    """Average ranks, so tied scores contribute half a concordance each."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    position = 0
    while position < len(values):
        end = position
        while end + 1 < len(values) and sorted_values[end + 1] == sorted_values[position]:
            end += 1
        ranks[order[position : end + 1]] = 0.5 * (position + end) + 1.0
        position = end + 1
    return ranks


def placement_values(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-record placement values for the positive and negative groups.

    Both groups are sorted first: the rank-difference identity that turns ranks into
    placements counts the members of the other group below each record, which only
    holds when the group is in ascending order.
    """
    positive = np.sort(scores[labels == 1])
    negative = np.sort(scores[labels == 0])
    if positive.size == 0 or negative.size == 0:
        raise ValueError("both classes must be present")
    combined = np.concatenate([positive, negative])
    ranks = midrank(combined)
    positive_ranks = ranks[: positive.size]
    negative_ranks = ranks[positive.size :]
    v10 = (positive_ranks - np.arange(1, positive.size + 1)) / negative.size
    v01 = 1.0 - (negative_ranks - np.arange(1, negative.size + 1)) / positive.size
    return v10, v01


def auc_variance(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """AUC and its DeLong variance."""
    positive = int((labels == 1).sum())
    negative = int((labels == 0).sum())
    if positive == 0 or negative == 0:
        raise ValueError("both classes must be present")
    v10, v01 = placement_values(scores, labels)
    auc = float(v10.mean())
    if positive < 2 or negative < 2:
        # A single record in a class leaves the placement variance undefined; the
        # point estimate is still the exact area, so only the variance is lost.
        return auc, float("nan")
    var = float(v10.var(ddof=1) / positive + v01.var(ddof=1) / negative)
    return auc, max(var, 0.0)


def delong_auc_ci(scores: np.ndarray, labels: np.ndarray, alpha: float = 0.05) -> DeLongResult:
    """AUC with its two-sided ``1 - alpha`` DeLong interval."""
    auc, variance = auc_variance(scores, labels)
    if not np.isfinite(variance):
        return DeLongResult(
            auc=auc,
            variance=float("nan"),
            standard_error=float("nan"),
            low=float("nan"),
            high=float("nan"),
            n_positive=int((labels == 1).sum()),
            n_negative=int((labels == 0).sum()),
        )
    se = float(np.sqrt(variance))
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    return DeLongResult(
        auc=auc,
        variance=variance,
        standard_error=se,
        low=max(0.0, auc - z * se),
        high=min(1.0, auc + z * se),
        n_positive=int((labels == 1).sum()),
        n_negative=int((labels == 0).sum()),
    )


def macro_auc_ci(scores: np.ndarray, labels: np.ndarray, n_classes: int, alpha: float = 0.05) -> DeLongResult:
    """Macro-averaged one-vs-rest AUC with an averaged DeLong interval.

    The per-class variances are averaged with equal weights, which is the interval
    the macro point estimate corresponds to; the between-class covariance is not
    estimated because the classes are not independent samples of the same estimand.
    """
    aucs: list[float] = []
    variances: list[float] = []
    for category in range(n_classes):
        binary = (labels == category).astype(np.int64)
        if binary.sum() < 2 or binary.sum() > len(binary) - 2:
            continue
        auc, variance = auc_variance(scores[:, category], binary)
        if not np.isfinite(variance):
            continue
        aucs.append(auc)
        variances.append(variance)
    if not aucs:
        return DeLongResult(float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), 0, 0)
    mean_auc = float(np.mean(aucs))
    se = float(np.sqrt(np.mean(variances) / len(variances)))
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    return DeLongResult(
        auc=mean_auc,
        variance=se**2,
        standard_error=se,
        low=max(0.0, mean_auc - z * se),
        high=min(1.0, mean_auc + z * se),
        n_positive=int((labels > 0).sum()),
        n_negative=int((labels == 0).sum()),
    )


def auc_difference_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Paired AUC difference with a placement-value covariance interval.

    Both arms are scored on the same records, so the covariance between their
    placement values has to enter the variance of the difference; ignoring it would
    overstate the interval and hide a real difference.
    """
    v10_a, v01_a = placement_values(scores_a, labels)
    v10_b, v01_b = placement_values(scores_b, labels)
    positive = int((labels == 1).sum())
    negative = int((labels == 0).sum())
    auc_a = float(v10_a.mean())
    auc_b = float(v10_b.mean())
    var_a = v10_a.var(ddof=1) / positive + v01_a.var(ddof=1) / negative
    var_b = v10_b.var(ddof=1) / positive + v01_b.var(ddof=1) / negative
    covariance = float(np.cov(v10_a, v10_b, ddof=1)[0, 1] / positive + np.cov(v01_a, v01_b, ddof=1)[0, 1] / negative)
    variance = max(var_a + var_b - 2.0 * covariance, 0.0)
    difference = auc_b - auc_a
    se = float(np.sqrt(variance))
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    return {
        "difference": difference,
        "standard_error": se,
        "low": difference - z * se,
        "high": difference + z * se,
        "z": difference / se if se > 1e-12 else 0.0,
        "p_value": float(2.0 * (1.0 - stats.norm.cdf(abs(difference / se)))) if se > 1e-12 else 1.0,
    }
