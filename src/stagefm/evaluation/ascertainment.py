"""Nodal ascertainment analysis.

The pooled statistics do not say where the model fails, so the nodal error is
decomposed per site and then related to how completely each site samples the nodal
basin. Two associations are computed: nodal error against the site's median
harvested node count, and the same against the tumour error, which is the control
that makes the first informative. Case volume and scanner vendor are checked as
alternative explanations. The between-site variance of the per-examination nodal
error is then recomputed on the subset whose yield reaches the adequacy threshold,
which is the quantity the manuscript reports as a 71% reduction.

Ref: Methods Sec. 2.4; Fig. 2; Algorithm 4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..stats.bootstrap import bootstrap_variance_components
from ..stats.correlation import pearson
from ..stats.icc import decompose, variance_reduction
from .loop import PredictionBundle
from .per_site import nodal_error_by_site, tumour_error_by_site


@dataclass(frozen=True)
class AscertainmentReport:
    """Everything the ascertainment analysis produces."""

    median_nodes: dict[str, int]
    nodal_error: dict[str, float]
    tumour_error: dict[str, float]
    nodal_misclassification: dict[str, float]
    case_volume: dict[str, int]
    nodal_correlation: dict[str, float]
    tumour_correlation: dict[str, float]
    volume_correlation: dict[str, float]
    vendor_groups: dict[str, float]
    full_decomposition: dict[str, float]
    restricted_decomposition: dict[str, float]
    restricted_count: int
    retained_share: float
    variance_reduction: float

    def as_dict(self) -> dict[str, object]:
        return {
            "median_nodes": self.median_nodes,
            "nodal_error": self.nodal_error,
            "tumour_error": self.tumour_error,
            "nodal_misclassification": self.nodal_misclassification,
            "case_volume": self.case_volume,
            "nodal_correlation": self.nodal_correlation,
            "tumour_correlation": self.tumour_correlation,
            "volume_correlation": self.volume_correlation,
            "vendor_groups": self.vendor_groups,
            "full_decomposition": self.full_decomposition,
            "restricted_decomposition": self.restricted_decomposition,
            "restricted_count": self.restricted_count,
            "retained_share": self.retained_share,
            "variance_reduction": self.variance_reduction,
        }


def nodal_error_indicator(bundle: PredictionBundle) -> np.ndarray:
    """Per-examination nodal disagreement: decoded category against the reference."""
    predicted = np.argmax(bundle.n_prob, axis=-1)
    indicator: np.ndarray = (predicted != bundle.labels["n"]).astype(np.float64)
    return indicator


def tumour_error_indicator(bundle: PredictionBundle) -> np.ndarray:
    """Per-examination tumour disagreement: decoded category against the reference."""
    predicted = np.argmax(bundle.t_prob, axis=-1)
    indicator: np.ndarray = (predicted != bundle.labels["t"]).astype(np.float64)
    return indicator


def vendor_groups_from(columns: dict[str, np.ndarray]) -> dict[str, list[int]]:
    """Positions grouped by scanner vendor."""
    vendors = columns["scanner_vendor"]
    grouped: dict[str, list[int]] = {}
    for position, vendor in enumerate(vendors.tolist()):
        grouped.setdefault(str(vendor), []).append(position)
    return grouped


def ascertainment_report(
    bundle: PredictionBundle,
    columns: dict[str, np.ndarray],
    median_nodes: dict[str, int],
    cutoff: int,
) -> AscertainmentReport:
    """Run the full ascertainment analysis on an external-layer bundle."""
    sites = sorted(set(bundle.sites))
    nodal = nodal_error_by_site(bundle)
    tumour = tumour_error_by_site(bundle)
    from .per_site import nodal_misclassification_by_site

    misclassification = nodal_misclassification_by_site(bundle)
    volume = {site: int(sum(1 for name in bundle.sites if name == site)) for site in sites}
    node_counts = np.array([median_nodes.get(site, 0) for site in sites], dtype=np.float64)
    nodal_values = np.array([nodal[site] for site in sites], dtype=np.float64)
    tumour_values = np.array([tumour[site] for site in sites], dtype=np.float64)
    volume_values = np.array([volume[site] for site in sites], dtype=np.float64)

    indicator = nodal_error_indicator(bundle)
    site_array = np.array(bundle.sites)
    full = decompose(indicator, site_array)
    node_yield = bundle.harvested_nodes
    adequate = node_yield >= cutoff
    if adequate.sum() >= 2 and len(set(site_array[adequate].tolist())) >= 2:
        restricted = decompose(indicator[adequate], site_array[adequate])
    else:
        restricted = full
    vendor_groups = vendor_groups_from(columns)
    vendor_errors: dict[str, float] = {}
    for vendor, positions in vendor_groups.items():
        selector = np.zeros(len(bundle), dtype=bool)
        selector[positions] = True
        if selector.any():
            vendor_errors[vendor] = float(indicator[selector].mean())

    return AscertainmentReport(
        median_nodes={site: int(median_nodes.get(site, 0)) for site in sites},
        nodal_error=nodal,
        tumour_error=tumour,
        nodal_misclassification=misclassification,
        case_volume=volume,
        nodal_correlation=pearson(node_counts, nodal_values),
        tumour_correlation=pearson(node_counts, tumour_values),
        volume_correlation=pearson(volume_values, nodal_values),
        vendor_groups=vendor_errors,
        full_decomposition=full.as_dict(),
        restricted_decomposition=restricted.as_dict(),
        restricted_count=int(adequate.sum()),
        retained_share=float(adequate.mean()) if len(adequate) else float("nan"),
        variance_reduction=variance_reduction(full.between_site_variance, restricted.between_site_variance),
    )


def bootstrap_report(
    bundle: PredictionBundle,
    cutoff: int,
    replicates: int = 2000,
    seed: int = 0,
) -> dict[str, object]:
    """Bootstrap interval for the between-site variance of the nodal error."""
    indicator = nodal_error_indicator(bundle)
    site_array = np.array(bundle.sites)
    full = bootstrap_variance_components(indicator, site_array, resamples=replicates, seed=seed)
    adequate = bundle.harvested_nodes >= cutoff
    if adequate.sum() >= 2 and len(set(site_array[adequate].tolist())) >= 2:
        restricted = bootstrap_variance_components(indicator[adequate], site_array[adequate], resamples=replicates, seed=seed + 1)
    else:
        restricted = full
    return {"full": full, "restricted": restricted}


def vendor_effect(indicator_by_vendor: dict[str, np.ndarray]) -> dict[str, Any]:
    """One-way comparison of the per-examination nodal error across vendors.

    A Kruskal-Wallis test is used because the error indicator is binary and the
    group sizes are unequal; the p-value answers whether the acquisition vendor
    explains any of the nodal error, which the manuscript reports as not significant.

    Ref: Methods Sec. 2.4 (scanner manufacturer shows no comparable relationship).
    """
    from scipy import stats

    groups = [values for values in indicator_by_vendor.values() if values.size > 0]
    if len(groups) < 2:
        return {"p_value": float("nan"), "groups": len(groups)}
    result = stats.kruskal(*groups)
    return {"p_value": float(result.pvalue), "statistic": float(result.statistic), "groups": len(groups)}
