"""Feasibility metric: the share of predictions forming an achievable stage combination.

Feasibility is reported alongside the endpoint and never on its own, because the
post-hoc projection reaches 100% feasibility while retaining a boundary discordance
of its own; the metric is only informative together with the decision column.

Ref: Table 3 (feasibility column); Sec. 3 (failing safely).
"""

from __future__ import annotations

import numpy as np

from ..data.schema import ALL_TRIPLES
from ..data.staging import AchievableSet


def feasibility_rate(columns: np.ndarray, achievable: AchievableSet) -> float:
    """Share of predicted stage columns that lie in the achievable set."""
    columns = np.asarray(columns, dtype=np.int64)
    if columns.size == 0:
        return float("nan")
    mask = np.asarray(achievable.mask, dtype=bool)
    return float(mask[np.clip(columns, 0, mask.shape[0] - 1)].mean())


def infeasible_examples(columns: np.ndarray, achievable: AchievableSet, limit: int = 5) -> list[str]:
    """Clinical shorthand of the first few infeasible predictions, for the report."""
    columns = np.asarray(columns, dtype=np.int64)
    labels = [ALL_TRIPLES[int(column)].label() for column in columns if not achievable.contains(ALL_TRIPLES[int(column)])]
    return labels[:limit]


def feasibility_breakdown(columns: np.ndarray, achievable: AchievableSet, sites: list[str]) -> dict[str, object]:
    """Pooled and per-site feasibility, in percent."""
    columns = np.asarray(columns, dtype=np.int64)
    site_array = np.asarray(sites)
    per_site: dict[str, float] = {}
    for site in sorted(set(sites)):
        selector = site_array == site
        per_site[site] = float(100.0 * feasibility_rate(columns[selector], achievable))
    return {"pooled": float(100.0 * feasibility_rate(columns, achievable)), "per_site": per_site}
