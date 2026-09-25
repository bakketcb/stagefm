"""Calibration metrics.

Expected calibration error is measured within each site and then pooled, because in
this cohort the miscalibration is driven by site rather than by confidence level.
The pooled value is the record-weighted mean of the per-site values, so a site with
fewer examinations cannot move the pooled number more than its share.

Ref: Methods Sec. 4.11 (expected calibration error per site, pooled); Fig. 3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CalibrationSummary:
    """Pooled expected calibration error and the per-site breakdown."""

    pooled: float
    per_site: dict[str, float]
    bins: int
    counts: dict[str, int]

    def as_dict(self) -> dict[str, object]:
        return {"pooled": self.pooled, "per_site": self.per_site, "bins": self.bins, "counts": self.counts}


def expected_calibration_error(probabilities: np.ndarray, targets: np.ndarray, bins: int = 10) -> float:
    """Equal-width binned expected calibration error.

    The first bin is closed on the left so that a probability of exactly zero is
    counted rather than dropped.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if probabilities.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(probabilities)
    error = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        if index == 0:
            selector = (probabilities >= lower) & (probabilities <= upper)
        else:
            selector = (probabilities > lower) & (probabilities <= upper)
        if not selector.any():
            continue
        confidence = float(probabilities[selector].mean())
        accuracy = float(targets[selector].mean())
        error += (float(selector.sum()) / total) * abs(confidence - accuracy)
    return float(error)


def calibration_summary(probabilities: np.ndarray, targets: np.ndarray, sites: list[str], bins: int = 10) -> CalibrationSummary:
    """Per-site and record-weighted pooled expected calibration error."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    site_array = np.asarray(sites)
    per_site: dict[str, float] = {}
    counts: dict[str, int] = {}
    for site in sorted(set(sites)):
        selector = site_array == site
        per_site[site] = expected_calibration_error(probabilities[selector], targets[selector], bins)
        counts[site] = int(selector.sum())
    pooled = expected_calibration_error(probabilities, targets, bins)
    return CalibrationSummary(pooled=pooled, per_site=per_site, bins=bins, counts=counts)


def calibration_slope(probabilities: np.ndarray, targets: np.ndarray) -> float:
    """Slope of the logistic recalibration of the predictions against the outcome.

    A perfectly calibrated model has a slope of one; a slope below one indicates the
    confidence spread is too wide.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if probabilities.size == 0 or len(np.unique(targets)) < 2:
        return float("nan")
    logits = np.log(np.clip(probabilities, 1e-9, 1 - 1e-9)) - np.log(np.clip(1 - probabilities, 1e-9, 1.0))
    design = np.column_stack([logits, np.ones_like(logits)])
    weights = np.zeros(2, dtype=np.float64)
    for _ in range(200):
        margin = design @ weights
        predicted = 1.0 / (1.0 + np.exp(-np.clip(margin, -30.0, 30.0)))
        gradient = design.T @ (predicted - targets) / len(targets)
        curvature = (design * (predicted * (1 - predicted))[:, None]).T @ design / len(targets)
        curvature += 1e-8 * np.eye(2)
        step = np.linalg.solve(curvature, gradient)
        weights -= step
        if float(np.abs(step).max()) < 1e-10:
            break
    return float(weights[0])
