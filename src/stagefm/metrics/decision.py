"""Decision metrics: treatment-boundary discordance, net benefit and reclassification.

Treatment-boundary discordance is the primary endpoint: the share of examinations on
which the predicted stage implies a management category different from the one the
pathological stage supports. It is read off the same boundary rule used to build the
labels, so it cannot drift from the definition.

Net benefit follows the standard decision-curve form, and the net reclassification
improvement is computed against a fixed reference arm on the clinically relevant
threshold range.

Ref: Methods Sec. 4.1 (the endpoint definition); Sec. 4.11 (net benefit and decision
curve analysis); Fig. 1 and Fig. 4.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data.schema import ALL_TRIPLES, TreatmentCategory


@dataclass(frozen=True)
class DiscordanceSummary:
    """The primary endpoint with its per-site breakdown."""

    pooled: float
    per_site: dict[str, float]
    per_site_counts: dict[str, int]

    def as_dict(self) -> dict[str, object]:
        return {"pooled": self.pooled, "per_site": self.per_site, "per_site_counts": self.per_site_counts}


def category_of_column(column: int) -> TreatmentCategory:
    """Management category implied by a flat stage column."""
    from ..data.staging import treatment_category

    return treatment_category(ALL_TRIPLES[column])


_CATEGORY_MAPPING = np.array(
    [
        {TreatmentCategory.SURGERY_FIRST: 0, TreatmentCategory.PERIOPERATIVE: 1, TreatmentCategory.SYSTEMIC: 2}[category_of_column(column)]
        for column in range(len(ALL_TRIPLES))
    ],
    dtype=np.int64,
)


def discordance_indicator(predicted_columns: np.ndarray, reference_columns: np.ndarray) -> np.ndarray:
    """Per-examination indicator of a category mismatch against the reference."""
    predicted = np.asarray(predicted_columns, dtype=np.int64)
    reference = np.asarray(reference_columns, dtype=np.int64)
    if predicted.size == 0:
        return np.zeros(0, dtype=np.int64)
    indicator: np.ndarray = (_CATEGORY_MAPPING[predicted] != _CATEGORY_MAPPING[reference]).astype(np.int64)
    return indicator


def boundary_discordance(predicted_columns: np.ndarray, reference_columns: np.ndarray) -> float:
    """Share of examinations with a category mismatch against the reference."""
    indicator = discordance_indicator(predicted_columns, reference_columns)
    if indicator.size == 0:
        return float("nan")
    return float(indicator.mean())


def discordance_summary(predicted_columns: np.ndarray, reference_columns: np.ndarray, sites: list[str]) -> DiscordanceSummary:
    """Pooled and per-site treatment-boundary discordance, in percent."""
    indicator = discordance_indicator(predicted_columns, reference_columns)
    site_array = np.asarray(sites)
    per_site: dict[str, float] = {}
    counts: dict[str, int] = {}
    for site in sorted(set(sites)):
        selector = site_array == site
        counts[site] = int(selector.sum())
        per_site[site] = float(100.0 * indicator[selector].mean()) if selector.any() else float("nan")
    pooled = float(100.0 * indicator.mean()) if indicator.size else float("nan")
    return DiscordanceSummary(pooled=pooled, per_site=per_site, per_site_counts=counts)


def net_benefit(positive: np.ndarray, predicted_positive: np.ndarray, threshold: float) -> float:
    """Standard net benefit at a decision threshold.

    ``net benefit = TP / n - FP / n * threshold / (1 - threshold)``.
    """
    positive = np.asarray(positive, dtype=np.int64)
    predicted_positive = np.asarray(predicted_positive, dtype=np.int64)
    total = len(positive)
    if total == 0:
        return float("nan")
    if threshold <= 0.0 or threshold >= 1.0:
        raise ValueError("decision threshold must lie strictly between 0 and 1")
    true_positive = float(((predicted_positive == 1) & (positive == 1)).sum())
    false_positive = float(((predicted_positive == 1) & (positive == 0)).sum())
    odds = threshold / (1.0 - threshold)
    return float(true_positive / total - false_positive / total * odds)


def decision_curve(positive: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray) -> dict[str, np.ndarray]:
    """Net benefit for the model, treat-all and treat-none across a threshold range."""
    positive = np.asarray(positive, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    prevalence = float(positive.mean()) if positive.size else float("nan")
    model = np.array([net_benefit(positive, (probabilities >= t).astype(np.int64), float(t)) for t in thresholds])
    treat_all = np.array([prevalence - (1.0 - prevalence) * (t / (1.0 - t)) for t in thresholds])
    treat_none = np.zeros_like(np.asarray(thresholds, dtype=np.float64))
    return {"threshold": np.asarray(thresholds, dtype=np.float64), "model": model, "treat_all": treat_all, "treat_none": treat_none}


def net_reclassification_improvement(
    reference_positive: np.ndarray,
    candidate_positive: np.ndarray,
    positive: np.ndarray,
) -> float:
    """Continuous net reclassification improvement of a candidate over a reference arm.

    Events reclassified upward and non-events reclassified downward count positively;
    the two directions are averaged so the measure is a rate rather than a count.
    """
    reference_positive = np.asarray(reference_positive, dtype=np.int64)
    candidate_positive = np.asarray(candidate_positive, dtype=np.int64)
    positive = np.asarray(positive, dtype=np.int64)
    events = positive == 1
    non_events = positive == 0
    if events.sum() == 0 or non_events.sum() == 0:
        return float("nan")
    up_events = float((candidate_positive[events] > reference_positive[events]).mean())
    down_non_events = float((candidate_positive[non_events] < reference_positive[non_events]).mean())
    return float(up_events + down_non_events)


def binary_nri(
    reference_positive: np.ndarray,
    candidate_positive: np.ndarray,
    positive: np.ndarray,
) -> dict[str, float]:
    """Event and non-event components of the binary reclassification improvement."""
    reference_positive = np.asarray(reference_positive, dtype=np.int64)
    candidate_positive = np.asarray(candidate_positive, dtype=np.int64)
    positive = np.asarray(positive, dtype=np.int64)
    events = positive == 1
    non_events = positive == 0
    event_component = float((candidate_positive[events] - reference_positive[events]).mean()) if events.any() else float("nan")
    non_event_component = float((reference_positive[non_events] - candidate_positive[non_events]).mean()) if non_events.any() else float("nan")
    return {
        "events": event_component,
        "non_events": non_event_component,
        "total": event_component + non_event_component,
    }
