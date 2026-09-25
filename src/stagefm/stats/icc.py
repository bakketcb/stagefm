"""Variance-component decomposition and the intraclass correlation.

The ascertainment analysis asks how much of the variation in nodal error sits
between sites rather than within them, so the decomposition is a one-way random
effects model on a per-examination error indicator with a site-level random
intercept, and the intraclass correlation is the between-site share. The reduction
in between-site variance after restricting to adequate nodal yield is the number the
manuscript reports.

Ref: Methods Sec. 2.4 (per-site N-axis error and the between-site variance
reduction); Sec. 4.11 (random-effects measurement); Algorithm 4.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VarianceComponents:
    """One-way random-effects decomposition of an outcome by group."""

    between_site_variance: float
    within_site_variance: float
    intraclass_correlation: float
    groups: int
    total: int
    mean_square_between: float
    mean_square_within: float

    def as_dict(self) -> dict[str, float]:
        return {
            "between_site_variance": self.between_site_variance,
            "within_site_variance": self.within_site_variance,
            "intraclass_correlation": self.intraclass_correlation,
            "mean_square_between": self.mean_square_between,
            "mean_square_within": self.mean_square_within,
        }


def random_effects_decomposition(values: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    """ANOVA mean squares and the derived variance components."""
    components = decompose(values, groups)
    return dict(components.as_dict())


def decompose(values: np.ndarray, groups: np.ndarray) -> VarianceComponents:
    """One-way random-effects decomposition with an unbalanced-design correction.

    The group-size correction ``n0`` is the effective per-group sample size; without
    it the between-site component would be inflated whenever the sites contribute
    unequal numbers of examinations, which is the normal case here.
    """
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    names = np.unique(groups)
    k = len(names)
    total = len(values)
    if k < 2 or total <= k:
        return VarianceComponents(
            between_site_variance=float("nan"),
            within_site_variance=float("nan"),
            intraclass_correlation=float("nan"),
            groups=k,
            total=total,
            mean_square_between=float("nan"),
            mean_square_within=float("nan"),
        )
    grand = float(values.mean())
    ss_between = 0.0
    ss_within = 0.0
    sum_squares_sizes = 0.0
    for name in names:
        block = values[groups == name]
        size = len(block)
        ss_between += size * (float(block.mean()) - grand) ** 2
        ss_within += float(((block - block.mean()) ** 2).sum())
        sum_squares_sizes += size**2
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (total - k)
    n0 = (total - sum_squares_sizes / total) / (k - 1)
    between = max((ms_between - ms_within) / n0, 0.0)
    within = ms_within
    divisor = between + within
    icc = between / divisor if divisor > 1e-15 else 0.0
    return VarianceComponents(
        between_site_variance=between,
        within_site_variance=within,
        intraclass_correlation=float(icc),
        groups=k,
        total=total,
        mean_square_between=ms_between,
        mean_square_within=ms_within,
    )


def variance_reduction(before: float, after: float) -> float:
    """Fractional reduction in a variance component, guarding against a zero baseline."""
    if not np.isfinite(before) or before <= 0.0:
        return float("nan")
    return float(1.0 - after / before)


def anova_table(values: np.ndarray, groups: np.ndarray) -> dict[str, dict[str, float]]:
    """Source-of-variation table with degrees of freedom, for the report."""
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    names = np.unique(groups)
    k = len(names)
    total = len(values)
    grand = float(values.mean())
    ss_between = float(sum(len(values[groups == name]) * (values[groups == name].mean() - grand) ** 2 for name in names))
    ss_within = float(sum(((values[groups == name] - values[groups == name].mean()) ** 2).sum() for name in names))
    ss_total = ss_between + ss_within
    return {
        "between": {"df": float(k - 1), "ss": ss_between, "ms": ss_between / (k - 1) if k > 1 else float("nan")},
        "within": {"df": float(total - k), "ss": ss_within, "ms": ss_within / (total - k) if total > k else float("nan")},
        "total": {"df": float(total - 1), "ss": ss_total, "ms": ss_total / (total - 1) if total > 1 else float("nan")},
    }
