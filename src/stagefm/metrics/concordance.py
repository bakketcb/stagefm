"""Concordance of the joint stage: exact-combination match rate and weighted kappa.

Concordance is the share of examinations on which all three axes match the
pathological stage exactly. Weighted kappa summarises the same agreement while
crediting near misses, using a linear weight so that an adjacent-category
disagreement costs half of a distant one; the manuscript does not name the weight
function, so the choice is recorded as an engineering default and both weightings
are computable here.

Ref: Methods Sec. 4.11 (weighted kappa summarising agreement with pathology); Table 3.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import cohen_kappa_score


def exact_combination_concordance(predicted_columns: np.ndarray, reference_columns: np.ndarray) -> float:
    """Share of examinations whose full stage combination matches the reference."""
    predicted = np.asarray(predicted_columns, dtype=np.int64)
    reference = np.asarray(reference_columns, dtype=np.int64)
    if predicted.size == 0:
        return float("nan")
    return float((predicted == reference).mean())


def axis_agreement(predicted: np.ndarray, reference: np.ndarray) -> float:
    """Share of examinations whose single axis matches."""
    predicted = np.asarray(predicted, dtype=np.int64)
    reference = np.asarray(reference, dtype=np.int64)
    if predicted.size == 0:
        return float("nan")
    return float((predicted == reference).mean())


def weighted_kappa(predicted: np.ndarray, reference: np.ndarray, weighting: str = "linear") -> float:
    """Weighted kappa between two ordinal label vectors."""
    predicted = np.asarray(predicted, dtype=np.int64)
    reference = np.asarray(reference, dtype=np.int64)
    if predicted.size == 0 or len(np.unique(predicted)) < 2 and len(np.unique(reference)) < 2:
        return float("nan")
    if np.array_equal(predicted, reference):
        return 1.0
    try:
        return float(cohen_kappa_score(reference, predicted, weights=weighting))
    except ValueError:
        return float("nan")


def ordinal_distance_agreement(predicted_columns: np.ndarray, reference_columns: np.ndarray) -> float:
    """Joint agreement credited by ordinal distance rather than exact match.

    The distance between two joint stages is the sum of the per-axis ordinal
    distances, normalised by the largest possible such sum, which keeps the measure
    consistent with the axis-wise disagreement the endpoint is defined on.
    """
    from ..data.schema import ALL_TRIPLES

    order = {axis: np.array([getattr(stage, axis) for stage in ALL_TRIPLES], dtype=np.float64) for axis in ("t", "n", "m")}
    predicted = np.asarray(predicted_columns, dtype=np.int64)
    reference = np.asarray(reference_columns, dtype=np.int64)
    if predicted.size == 0:
        return float("nan")
    total = 0.0
    for values in order.values():
        total += float(np.abs(values[predicted] - values[reference]).mean())
    maximum = 3.0 + 3.0 + 1.0
    return float(1.0 - total / maximum)
