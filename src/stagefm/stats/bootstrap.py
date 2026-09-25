"""Bootstrap intervals.

The primary endpoint's interval is a percentile bootstrap over examinations, and the
paired comparisons of two arms' discordance use a paired bootstrap over the same
records so that the correlation between the arms is carried through. The resample
count is fixed by the caller and reported with the result.

Ref: Methods Sec. 4.11 (bootstrap intervals for the endpoint and for subgroup
comparisons); Fig. 1 (interval on the primary endpoint).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    """Point estimate, percentile interval and replicate count."""

    estimate: float
    low: float
    high: float
    replicates: int
    standard_error: float

    @property
    def half_width(self) -> float:
        return 0.5 * (self.high - self.low)


def percentile_interval(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile interval of a replicate sample."""
    if values.size == 0:
        return float("nan"), float("nan")
    low = float(np.percentile(values, 100.0 * alpha / 2.0))
    high = float(np.percentile(values, 100.0 * (1.0 - alpha / 2.0)))
    return low, high


def bootstrap_statistic(
    statistic: Callable[[np.ndarray], float],
    count: int,
    resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> BootstrapResult:
    """Percentile bootstrap of a scalar statistic over ``count`` records."""
    if count <= 0:
        return BootstrapResult(estimate=float("nan"), low=float("nan"), high=float("nan"), replicates=0, standard_error=float("nan"))
    index = np.arange(count)
    rng = np.random.default_rng(seed)
    estimate = float(statistic(index))
    replicates = np.empty(resamples, dtype=np.float64)
    for position in range(resamples):
        draw = rng.integers(0, count, size=count)
        replicates[position] = statistic(draw)
    low, high = percentile_interval(replicates, alpha)
    return BootstrapResult(
        estimate=estimate,
        low=low,
        high=high,
        replicates=resamples,
        standard_error=float(np.nanstd(replicates, ddof=1)),
    )


def paired_bootstrap(
    statistic: Callable[[np.ndarray], float],
    count: int,
    resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> BootstrapResult:
    """Percentile bootstrap of a paired difference.

    ``statistic`` receives a positional index array sampled with replacement within
    the pair, so both arms of the comparison see exactly the same records in every
    replicate.
    """
    return bootstrap_statistic(statistic, count, resamples=resamples, alpha=alpha, seed=seed)


def bootstrap_variance_components(
    error_indicator: np.ndarray,
    site: np.ndarray,
    resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, object]:
    """Bootstrap the between-site and within-site variance of a binary error indicator.

    Resampling happens within each site stratum, then a one-way random-effects
    decomposition is fitted to the replicate. The intraclass correlation is the share
    of total variance that sits between sites, which is the quantity the
    ascertainment analysis tracks.

    Ref: Algorithm 4.
    """
    from .icc import random_effects_decomposition

    sites = np.unique(site)
    counts = {name: int((site == name).sum()) for name in sites}
    eligible = [name for name in sites if counts[name] >= 2]
    if len(eligible) < 2:
        return {
            "between_site_variance": float("nan"),
            "within_site_variance": float("nan"),
            "intraclass_correlation": float("nan"),
            "between_low": float("nan"),
            "between_high": float("nan"),
            "replicates": 0,
        }
    point = random_effects_decomposition(error_indicator, site)
    rng = np.random.default_rng(seed)
    between = np.empty(resamples, dtype=np.float64)
    within = np.empty(resamples, dtype=np.float64)
    icc = np.empty(resamples, dtype=np.float64)
    index_by_site = {name: np.flatnonzero(site == name) for name in eligible}
    for position in range(resamples):
        rows: list[np.ndarray] = []
        groups: list[np.ndarray] = []
        for name in eligible:
            pool = index_by_site[name]
            draw = rng.choice(pool, size=pool.size, replace=True)
            rows.append(error_indicator[draw])
            groups.append(np.full(pool.size, name))
        values = np.concatenate(rows)
        labels = np.concatenate(groups)
        decomposition = random_effects_decomposition(values, labels)
        between[position] = decomposition["between_site_variance"]
        within[position] = decomposition["within_site_variance"]
        icc[position] = decomposition["intraclass_correlation"]
    return {
        "between_site_variance": point["between_site_variance"],
        "within_site_variance": point["within_site_variance"],
        "intraclass_correlation": point["intraclass_correlation"],
        "between_low": float(np.nanpercentile(between, 100.0 * alpha / 2.0)),
        "between_high": float(np.nanpercentile(between, 100.0 * (1.0 - alpha / 2.0))),
        "icc_low": float(np.nanpercentile(icc, 100.0 * alpha / 2.0)),
        "icc_high": float(np.nanpercentile(icc, 100.0 * (1.0 - alpha / 2.0))),
        "replicates": resamples,
    }
