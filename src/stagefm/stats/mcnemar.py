"""McNemar test and the interval on a paired difference of proportions.

The load-bearing comparison in the manuscript is the reduction in treatment-boundary
discordance between the model and the unconstrained control measured on the same
examinations, so the test has to be the paired one and the interval has to come from
the discordant-pair counts rather than from an unpaired proportion interval.

Ref: Methods Sec. 4.11; Sec. 2.2 (McNemar p < 0.001 against the control arm).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class McNemarResult:
    """Discordant-pair counts, the exact p-value and the interval on the difference."""

    both_correct: int
    only_first: int
    only_second: int
    neither: int
    statistic: float
    statistic_continuity: float
    p_value: float
    p_value_exact: float
    difference: float
    low: float
    high: float

    @property
    def discordant(self) -> int:
        return self.only_first + self.only_second


def mcnemar(correct_first: np.ndarray, correct_second: np.ndarray, alpha: float = 0.05) -> McNemarResult:
    """Paired comparison of two binary correctness vectors.

    ``difference`` is the second arm's error rate minus the first's, so a negative
    value means the second arm is the safer one.
    """
    if correct_first.shape != correct_second.shape:
        raise ValueError("paired vectors must have the same shape")
    both = int(((correct_first == 1) & (correct_second == 1)).sum())
    only_first = int(((correct_first == 1) & (correct_second == 0)).sum())
    only_second = int(((correct_first == 0) & (correct_second == 1)).sum())
    neither = int(((correct_first == 0) & (correct_second == 0)).sum())
    discordant = only_first + only_second
    if discordant == 0:
        statistic = float("nan")
        p_value = 1.0
        p_exact = 1.0
    else:
        statistic = (abs(only_first - only_second) - 1) ** 2 / discordant
        p_value = float(1.0 - stats.chi2.cdf(statistic, df=1))
        p_exact = float(min(1.0, 2.0 * stats.binom.cdf(min(only_first, only_second), discordant, 0.5)))
    total = len(correct_first)
    difference = (only_second - only_first) / total if total else float("nan")
    # Wald interval on the difference of paired error rates, driven by the two
    # discordant cells rather than by the marginal error rates.
    if total == 0:
        low, high = float("nan"), float("nan")
    else:
        variance = max(only_first + only_second - (only_first - only_second) ** 2 / total, 0.0) / (total**2)
        se = math.sqrt(variance)
        z = float(stats.norm.ppf(1.0 - alpha / 2.0))
        low, high = difference - z * se, difference + z * se
    return McNemarResult(
        both_correct=both,
        only_first=only_first,
        only_second=only_second,
        neither=neither,
        statistic=statistic,
        statistic_continuity=float((max(abs(only_first - only_second) - 1, 0) ** 2 / discordant) if discordant else float("nan")),
        p_value=p_value,
        p_value_exact=p_exact,
        difference=difference,
        low=low,
        high=high,
    )


def paired_difference_ci(diff: np.ndarray, alpha: float = 0.05) -> dict[str, float]:
    """Normal-approximation interval on the mean of a per-record paired difference."""
    count = len(diff)
    if count < 2:
        return {"mean": float(diff.mean()) if count else float("nan"), "low": float("nan"), "high": float("nan")}
    mean = float(diff.mean())
    se = float(diff.std(ddof=1) / math.sqrt(count))
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    return {"mean": mean, "standard_error": se, "low": mean - z * se, "high": mean + z * se}
