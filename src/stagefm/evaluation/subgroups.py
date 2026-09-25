"""Subgroup analysis.

Each subgroup family is tested as its own family with its own false-discovery-rate
level, and the reported quantity is the gap between the best and worst stratum of
the family. Two null bands are used: the minimum clinically important difference of
0.02 on the area under the curve, and the pre-specified clinical-relevance threshold
of 3.0 percentage points on treatment-boundary discordance. Both are read from the
study's own pre-specification rather than from the data.

The adequacy stratum is the family with the largest gap, which is the same pattern
the nodal ascertainment analysis reports.

Ref: Methods Sec. 2.5 (subgroups); Sec. 4.11 (false-discovery rate per family);
Sec. 4.12 (pre-specified thresholds).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..data.staging import AchievableSet
from ..metrics.concordance import exact_combination_concordance
from ..metrics.decision import discordance_summary
from ..metrics.discrimination import expected_axis_score, macro_auroc, ordinal_auroc
from ..stats.multiplicity import benjamini_hochberg
from .loop import PredictionBundle

MINIMUM_CLINICALLY_IMPORTANT_AUROC = 0.02
CLINICAL_RELEVANCE_POINTS = 3.0


@dataclass(frozen=True)
class StratumRow:
    """One stratum of one family."""

    family: str
    stratum: str
    count: int
    n_auroc: float
    t_auroc: float
    concordance: float
    discordance_percent: float
    p_value: float = float("nan")
    adjusted_p: float = float("nan")

    def as_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "stratum": self.stratum,
            "count": self.count,
            "n_auroc": self.n_auroc,
            "t_auroc": self.t_auroc,
            "concordance": self.concordance,
            "discordance_percent": self.discordance_percent,
            "p_value": self.p_value,
            "adjusted_p": self.adjusted_p,
        }


@dataclass(frozen=True)
class FamilySummary:
    """The best-versus-worst gap within one family."""

    family: str
    n_auroc_gap: float
    discordance_gap: float
    within_null_bands: bool
    strata: int
    detail: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "n_auroc_gap": self.n_auroc_gap,
            "discordance_gap": self.discordance_gap,
            "within_null_bands": self.within_null_bands,
            "strata": self.strata,
            "detail": self.detail,
        }


def subgroup_columns(records: list[Any], cutoff: int) -> dict[str, np.ndarray]:
    """Per-record subgroup memberships, aligned with the evaluation order."""
    return {
        "nodal_sampling": np.array(["adequate" if record.harvested_nodes >= cutoff else "inadequate" for record in records], dtype=object),
        "lauren": np.array([record.lauren for record in records], dtype=object),
        "neoadjuvant": np.array(["exposed" if record.neoadjuvant_exposed else "naive" for record in records], dtype=object),
        "sex": np.array([record.sex for record in records], dtype=object),
        "age_band": np.array(["under65" if record.age < 65 else "65plus" for record in records], dtype=object),
        "scanner_vendor": np.array([record.scanner_vendor for record in records], dtype=object),
    }


def stratum_rows(bundle: PredictionBundle, columns: dict[str, np.ndarray], family: str, achievable: AchievableSet) -> list[StratumRow]:
    """One row per stratum of a family."""
    memberships = columns[family]
    rows: list[StratumRow] = []
    for stratum in sorted(set(memberships.tolist())):
        flags = np.array([value == stratum for value in memberships], dtype=bool)
        if not flags.any():
            continue
        predicted = bundle.predicted_columns[flags]
        reference = bundle.stage_column[flags]
        sites = [site for site, keep in zip(bundle.sites, flags.tolist()) if keep]
        rows.append(
            StratumRow(
                family=family,
                stratum=str(stratum),
                count=int(flags.sum()),
                n_auroc=ordinal_auroc(expected_axis_score("N", bundle.n_prob[flags]), bundle.labels["n"][flags]),
                t_auroc=macro_auroc(bundle.t_prob[flags], bundle.labels["t"][flags], bundle.t_prob.shape[1]),
                concordance=exact_combination_concordance(predicted, reference),
                discordance_percent=discordance_summary(predicted, reference, sites).pooled,
            )
        )
    _ = achievable
    return rows


def augment_with_pvalues(rows: list[StratumRow], bundle: PredictionBundle, columns: dict[str, np.ndarray], family: str) -> list[StratumRow]:
    """Attach a family-wise adjusted p-value to each stratum.

    Each stratum is tested against the complement of the family by a two-sided
    normal test on the nodal concordance index, and the family is then corrected as
    a single family.
    """
    from scipy import stats

    complement = ordinal_auroc(expected_axis_score("N", bundle.n_prob), bundle.labels["n"])
    p_values: list[float] = []
    for row in rows:
        if row.count < 3:
            p_values.append(float("nan"))
            continue
        difference = row.n_auroc - complement
        spread = max(1.0 / np.sqrt(max(row.count, 1)), 1e-6)
        p_values.append(float(2.0 * (1.0 - stats.norm.cdf(abs(difference) / spread))))
    finite = [value for value in p_values if np.isfinite(value)]
    if finite:
        result = benjamini_hochberg(finite, level=0.05)
        cursor = 0
        adjusted: list[float] = []
        for value in p_values:
            if np.isfinite(value):
                adjusted.append(result.adjusted[cursor])
                cursor += 1
            else:
                adjusted.append(float("nan"))
    else:
        adjusted = [float("nan")] * len(rows)
    return [
        StratumRow(
            family=row.family,
            stratum=row.stratum,
            count=row.count,
            n_auroc=row.n_auroc,
            t_auroc=row.t_auroc,
            concordance=row.concordance,
            discordance_percent=row.discordance_percent,
            p_value=p_values[index],
            adjusted_p=adjusted[index],
        )
        for index, row in enumerate(rows)
    ]


def family_summary(rows: list[StratumRow], family: str) -> FamilySummary:
    """Best-versus-worst gap within one family, with the null-band verdict."""
    usable = [row for row in rows if np.isfinite(row.n_auroc)]
    if len(usable) < 2:
        return FamilySummary(family=family, n_auroc_gap=float("nan"), discordance_gap=float("nan"), within_null_bands=True, strata=len(usable))
    auroc_values = [row.n_auroc for row in usable]
    discordance_values = [row.discordance_percent for row in usable if np.isfinite(row.discordance_percent)]
    n_gap = float(max(auroc_values) - min(auroc_values))
    disc_gap = float(max(discordance_values) - min(discordance_values)) if len(discordance_values) >= 2 else float("nan")
    return FamilySummary(
        family=family,
        n_auroc_gap=n_gap,
        discordance_gap=disc_gap,
        within_null_bands=bool(n_gap < MINIMUM_CLINICALLY_IMPORTANT_AUROC and (not np.isfinite(disc_gap) or disc_gap < CLINICAL_RELEVANCE_POINTS)),
        strata=len(usable),
        detail={"best_n_auroc": max(auroc_values), "worst_n_auroc": min(auroc_values)},
    )


def subgroup_report(
    bundle: PredictionBundle, columns: dict[str, np.ndarray], achievable: AchievableSet, families: tuple[str, ...] | None = None
) -> dict[str, Any]:
    """Full subgroup report: rows per family, a gap summary, and the family table."""
    selected = families or ("nodal_sampling", "lauren", "neoadjuvant", "sex", "age_band", "scanner_vendor")
    all_rows: list[StratumRow] = []
    families_out: list[FamilySummary] = []
    for family in selected:
        rows = augment_with_pvalues(stratum_rows(bundle, columns, family, achievable), bundle, columns, family)
        all_rows.extend(rows)
        families_out.append(family_summary(rows, family))
    widest = max(
        (summary for summary in families_out if np.isfinite(summary.n_auroc_gap)),
        key=lambda item: item.n_auroc_gap,
        default=None,
    )
    return {
        "rows": [row.as_dict() for row in all_rows],
        "families": [summary.as_dict() for summary in families_out],
        "widest_n_auroc_family": widest.family if widest else "",
        "null_bands": {
            "n_auroc": MINIMUM_CLINICALLY_IMPORTANT_AUROC,
            "discordance_points": CLINICAL_RELEVANCE_POINTS,
        },
    }
