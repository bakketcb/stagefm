"""Discrimination metrics for the three axes.

The T axis is a four-class macro-averaged area under the ROC curve, the M axis is
binary, and the N axis is ordinal: its one-dimensional summary is the concordance
index, computed as the sample-size weighted mean of the pairwise areas in which a
lower nodal category is expected to score below a higher one. Writing it as a
weighted mean of pairwise areas keeps it a genuine summary of the ordinal
discrimination rather than a binary threshold in disguise, and it reduces to the
ordinary area when only two categories are present.

Ref: Methods Sec. 4.11 (T four-class macro-averaged, N ordinal, M binary, summarised
with DeLong intervals).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..stats.delong import DeLongResult, auc_variance, delong_auc_ci, macro_auc_ci


@dataclass(frozen=True)
class AxisDiscrimination:
    """One axis's point estimate and interval."""

    axis: str
    auroc: float
    low: float
    high: float
    standard_error: float
    detail: dict[str, float] = field(default_factory=dict)


def binary_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve with midrank tie handling."""
    return auc_variance(np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.int64))[0]


def macro_auroc(scores: np.ndarray, labels: np.ndarray, n_classes: int) -> float:
    """Macro-averaged one-vs-rest area under the ROC curve."""
    return macro_auc_ci(np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.int64), n_classes).auc


def ordinal_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Concordance index for an ordinal axis, as a weighted mean of pairwise areas."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    categories = np.unique(labels)
    numerator = 0.0
    denominator = 0.0
    for position, lower in enumerate(categories):
        for higher in categories[position + 1 :]:
            low_scores = scores[labels == lower]
            high_scores = scores[labels == higher]
            weight = len(low_scores) * len(high_scores)
            if weight == 0:
                continue
            pairwise = _pairwise_area(low_scores, high_scores)
            numerator += pairwise * weight
            denominator += weight
    if denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def _pairwise_area(lower_scores: np.ndarray, higher_scores: np.ndarray) -> float:
    """Probability that a case from the higher category scores above one from the lower."""
    combined = np.concatenate([lower_scores, higher_scores])
    labels = np.concatenate([np.zeros(len(lower_scores), dtype=np.int64), np.ones(len(higher_scores), dtype=np.int64)])
    return auc_variance(combined, labels)[0]


def safe_binary_auroc(scores: np.ndarray, labels: np.ndarray, alpha: float = 0.05) -> DeLongResult:
    """DeLong interval that reports an undefined result when a class is absent.

    A site or a stratum can carry a single class on an axis; the diagnostic-accuracy
    reporting convention is to leave the value undefined rather than to raise or to
    substitute a default.
    """
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.size == 0 or len(np.unique(labels)) < 2:
        return DeLongResult(float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), int((labels == 1).sum()), int((labels == 0).sum()))
    return delong_auc_ci(scores, labels, alpha=alpha)


def ordinal_auroc_ci(scores: np.ndarray, labels: np.ndarray, alpha: float = 0.05, resamples: int = 2000, seed: int = 0) -> DeLongResult:
    """Concordance index with a bootstrap interval.

    There is no closed-form covariance for the multi-category concordance index, so
    the interval comes from resampling examinations; the closed form exists for each
    pairwise area and is used inside the point estimate.
    """
    from ..stats.bootstrap import bootstrap_statistic

    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    result = bootstrap_statistic(
        lambda index: ordinal_auroc(scores[index], labels[index]),
        count=len(labels),
        resamples=resamples,
        alpha=alpha,
        seed=seed,
    )
    return DeLongResult(
        auc=result.estimate,
        variance=result.standard_error**2,
        standard_error=result.standard_error,
        low=result.low,
        high=result.high,
        n_positive=int((labels > 0).sum()),
        n_negative=int((labels == 0).sum()),
    )


def axis_discrimination(
    t_scores: np.ndarray,
    n_scores: np.ndarray,
    m_scores: np.ndarray,
    labels: dict[str, np.ndarray],
    alpha: float = 0.05,
) -> dict[str, AxisDiscrimination]:
    """All three axes with their intervals."""
    t_result = macro_auc_ci(t_scores, labels["t"], n_classes=t_scores.shape[1], alpha=alpha)
    n_result = ordinal_auroc_ci(n_scores, labels["n"], alpha=alpha)
    m_result = delong_auc_ci(m_scores, labels["m"], alpha=alpha)
    return {
        "T": AxisDiscrimination("T", t_result.auc, t_result.low, t_result.high, t_result.standard_error),
        "N": AxisDiscrimination("N", n_result.auc, n_result.low, n_result.high, n_result.standard_error),
        "M": AxisDiscrimination("M", m_result.auc, m_result.low, m_result.high, m_result.standard_error),
    }


def expected_axis_score(axis: str, probabilities: np.ndarray) -> np.ndarray:
    """The scalar summary each axis is scored on.

    T uses the four class probabilities for the macro average, N uses the expected
    category so the ordinal structure is preserved, and M uses the positive-class
    probability.
    """
    if axis == "T":
        return probabilities
    if axis == "N":
        values = np.arange(probabilities.shape[1], dtype=np.float64)
        return probabilities @ values
    return probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]
