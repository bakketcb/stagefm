"""Multiplicity control and interaction ratios.

Subgroup and per-site test families each receive a false-discovery-rate level fixed
before the analysis, so the families are corrected separately rather than pooled.
The interaction ratio compares the joint effect of the two structural components
against the sum of their separate effects, on both the decision endpoint and the
exact-combination concordance.

Ref: Methods Sec. 4.11 (false-discovery rate assigned per family); Sec. 2.3
(interaction ratio 1.30 on the decision endpoint and 0.87 on concordance).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BHResult:
    """Benjamini-Hochberg adjusted p-values and the rejected set."""

    rejected: list[int]
    adjusted: list[float]
    level: float
    family_size: int

    def as_dict(self) -> dict[str, object]:
        return {"rejected": self.rejected, "adjusted": self.adjusted, "level": self.level, "family_size": self.family_size}


def benjamini_hochberg(p_values: list[float], level: float = 0.05) -> BHResult:
    """Benjamini-Hochberg step-up procedure.

    The adjusted value of the i-th smallest p-value is the running minimum of
    ``p * m / i`` from the largest down, which keeps the adjusted sequence monotone
    in the original order.
    """
    count = len(p_values)
    if count == 0:
        return BHResult(rejected=[], adjusted=[], level=level, family_size=0)
    order = np.argsort(np.asarray(p_values, dtype=np.float64), kind="mergesort")
    sorted_p = np.asarray(p_values, dtype=np.float64)[order]
    ranks = np.arange(1, count + 1, dtype=np.float64)
    scaled = sorted_p * count / ranks
    monotone = np.minimum.accumulate(scaled[::-1])[::-1]
    monotone = np.clip(monotone, 0.0, 1.0)
    adjusted = np.empty(count, dtype=np.float64)
    adjusted[order] = monotone
    rejected = [int(order[position]) for position in range(count) if sorted_p[position] <= level * ranks[position] / count]
    return BHResult(rejected=rejected, adjusted=[float(value) for value in adjusted], level=level, family_size=count)


def interaction_ratio(joint: float, separate_first: float, separate_second: float) -> float:
    """Joint effect divided by the sum of the separate effects.

    A value above one means the two components act super-additively on the outcome
    they are measured against; below one, sub-additively. The reference
    configuration is the one in which both components are absent, so each argument is
    the *improvement* over that reference.
    """
    denominator = separate_first + separate_second
    if abs(denominator) < 1e-15:
        return float("nan")
    return float(joint / denominator)


@dataclass(frozen=True)
class InteractionDecomposition:
    """The four arms of a two-component interaction on one outcome."""

    reference: float
    without_first: float
    without_second: float
    full: float
    direction: str = "improvement"

    @property
    def separate_first(self) -> float:
        """Effect of the first component with the second absent, against the reference."""
        return self._gap(self.without_second, self.reference)

    @property
    def separate_second(self) -> float:
        """Effect of the second component with the first absent, against the reference."""
        return self._gap(self.without_first, self.reference)

    @property
    def joint(self) -> float:
        return self._gap(self.full, self.reference)

    @property
    def ratio(self) -> float:
        return interaction_ratio(self.joint, self.separate_first, self.separate_second)

    def _gap(self, value: float, reference: float) -> float:
        return float(abs(value - reference))

    def as_dict(self) -> dict[str, float]:
        return {
            "reference": self.reference,
            "without_first": self.without_first,
            "without_second": self.without_second,
            "full": self.full,
            "separate_first": self.separate_first,
            "separate_second": self.separate_second,
            "joint": self.joint,
            "ratio": self.ratio,
        }


def standardised_difference(before: float, after: float, spread: float) -> float:
    """Difference standardised by a spread, for cross-outcome comparability."""
    if spread <= 0.0:
        return float("nan")
    return float((after - before) / spread)
