"""Per-site evaluation.

Every reported quantity is computed within each site and then pooled, because the
pooled number hides which site carries the failure. The T axis is expected to be
stable across sites, the nodal axis is not, and the treatment-boundary discordance
tracks the nodal axis rather than the tumour axis.

Ref: Methods Sec. 2.4 (per-site consistency); Table 3 (pooled values); Fig. 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..data.staging import AchievableSet
from ..metrics.calibration import expected_calibration_error
from ..metrics.concordance import exact_combination_concordance, weighted_kappa
from ..metrics.decision import discordance_summary
from ..metrics.discrimination import expected_axis_score, macro_auroc, ordinal_auroc, safe_binary_auroc
from ..metrics.feasibility import feasibility_rate
from .loop import PredictionBundle


@dataclass(frozen=True)
class SiteReport:
    """Metrics for one site."""

    site: str
    region: str
    count: int
    t_auroc: float
    n_auroc: float
    m_auroc: float
    concordance: float
    weighted_kappa: float
    discordance_percent: float
    expected_calibration_error: float
    feasibility_percent: float

    def as_dict(self) -> dict[str, object]:
        return {
            "site": self.site,
            "region": self.region,
            "count": self.count,
            "t_auroc": self.t_auroc,
            "n_auroc": self.n_auroc,
            "m_auroc": self.m_auroc,
            "concordance": self.concordance,
            "weighted_kappa": self.weighted_kappa,
            "discordance_percent": self.discordance_percent,
            "expected_calibration_error": self.expected_calibration_error,
            "feasibility_percent": self.feasibility_percent,
        }


@dataclass(frozen=True)
class PerSiteReport:
    """Per-site rows plus the pooled summary and the spread across sites."""

    rows: list[SiteReport] = field(default_factory=list)
    pooled: dict[str, float] = field(default_factory=dict)

    def spread(self, metric: str) -> dict[str, float | str]:
        """Minimum, maximum and spread of one metric across sites."""
        values = [float(getattr(row, metric)) for row in self.rows if np.isfinite(float(getattr(row, metric)))]
        if not values:
            return {"min": float("nan"), "max": float("nan"), "spread": float("nan"), "site_min": "", "site_max": ""}
        minimum = min(values)
        maximum = max(values)
        return {
            "min": minimum,
            "max": maximum,
            "spread": maximum - minimum,
            "site_min": self.rows[values.index(minimum)].site,
            "site_max": self.rows[values.index(maximum)].site,
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "rows": [row.as_dict() for row in self.rows],
            "pooled": self.pooled,
            "t_auroc_spread": self.spread("t_auroc"),
            "n_auroc_spread": self.spread("n_auroc"),
            "discordance_spread": self.spread("discordance_percent"),
        }


def region_of(site: str, regions: dict[str, str]) -> str:
    return regions.get(site, "unknown")


def per_site_report(
    bundle: PredictionBundle,
    achievable: AchievableSet,
    regions: dict[str, str],
    include_kappa: bool = True,
) -> PerSiteReport:
    """Compute the report for every site present in the bundle."""
    sites = sorted(set(bundle.sites))
    rows: list[SiteReport] = []
    for site in sites:
        flags = np.array([name == site for name in bundle.sites], dtype=bool)
        block = bundle
        t_scores = block.t_prob[flags]
        n_scores = expected_axis_score("N", block.n_prob[flags])
        m_scores = block.m_prob[flags][:, 1] if block.m_prob.shape[1] > 1 else block.m_prob[flags][:, 0]
        labels = {"t": block.labels["t"][flags], "n": block.labels["n"][flags], "m": block.labels["m"][flags]}
        predicted = block.predicted_columns[flags]
        reference = block.stage_column[flags]
        m_binary = safe_binary_auroc(m_scores, labels["m"])
        rows.append(
            SiteReport(
                site=site,
                region=region_of(site, regions),
                count=int(flags.sum()),
                t_auroc=macro_auroc(t_scores, labels["t"], t_scores.shape[1]),
                n_auroc=ordinal_auroc(n_scores, labels["n"]),
                m_auroc=m_binary.auc,
                concordance=exact_combination_concordance(predicted, reference),
                weighted_kappa=weighted_kappa(predicted, reference) if include_kappa else float("nan"),
                discordance_percent=discordance_summary(predicted, reference, [site] * int(flags.sum())).pooled,
                expected_calibration_error=expected_calibration_error(_positive_probability(block)[flags], labels["m"]),
                feasibility_percent=float(100.0 * feasibility_rate(predicted, achievable)),
            )
        )
    return PerSiteReport(rows=rows, pooled=pooled_summary(bundle, achievable, include_kappa=include_kappa))


def _positive_probability(bundle: PredictionBundle) -> np.ndarray:
    return bundle.m_prob[:, 1] if bundle.m_prob.shape[1] > 1 else bundle.m_prob[:, 0]


def pooled_summary(bundle: PredictionBundle, achievable: AchievableSet, include_kappa: bool = True) -> dict[str, float]:
    """The pooled value of every metric, over all records in the bundle."""
    t_scores = bundle.t_prob
    n_scores = expected_axis_score("N", bundle.n_prob)
    m_scores = _positive_probability(bundle)
    predicted = bundle.predicted_columns
    reference = bundle.stage_column
    return {
        "t_auroc": macro_auroc(t_scores, bundle.labels["t"], t_scores.shape[1]),
        "n_auroc": ordinal_auroc(n_scores, bundle.labels["n"]),
        "m_auroc": safe_binary_auroc(m_scores, bundle.labels["m"]).auc,
        "concordance": exact_combination_concordance(predicted, reference),
        "weighted_kappa": weighted_kappa(predicted, reference) if include_kappa else float("nan"),
        "discordance_percent": discordance_summary(predicted, reference, list(bundle.sites)).pooled,
        "expected_calibration_error": expected_calibration_error(_positive_probability(bundle), bundle.labels["m"]),
        "feasibility_percent": float(100.0 * feasibility_rate(predicted, achievable)),
        "count": float(len(bundle)),
    }


def nodal_error_by_site(bundle: PredictionBundle) -> dict[str, float]:
    """Per-site nodal-axis error, defined as one minus the nodal-area under the curve."""
    errors: dict[str, float] = {}
    for site in sorted(set(bundle.sites)):
        flags = np.array([name == site for name in bundle.sites], dtype=bool)
        errors[site] = float(1.0 - ordinal_auroc(expected_axis_score("N", bundle.n_prob[flags]), bundle.labels["n"][flags]))
    return errors


def tumour_error_by_site(bundle: PredictionBundle) -> dict[str, float]:
    """Per-site tumour-axis error, defined as one minus the macro tumour-area."""
    errors: dict[str, float] = {}
    for site in sorted(set(bundle.sites)):
        flags = np.array([name == site for name in bundle.sites], dtype=bool)
        errors[site] = float(1.0 - macro_auroc(bundle.t_prob[flags], bundle.labels["t"][flags], bundle.t_prob.shape[1]))
    return errors


def nodal_misclassification_by_site(bundle: PredictionBundle) -> dict[str, float]:
    """Per-site nodal disagreement rate on the decoded category.

    Reported alongside the area-based error so a reader can see that the
    ascertainment pattern is not an artefact of the summary chosen.
    """
    rates: dict[str, float] = {}
    for site in sorted(set(bundle.sites)):
        flags = np.array([name == site for name in bundle.sites], dtype=bool)
        predicted_n = np.argmax(bundle.n_prob[flags], axis=-1)
        rates[site] = float((predicted_n != bundle.labels["n"][flags]).mean())
    return rates
