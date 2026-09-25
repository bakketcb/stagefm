"""Site-conditional calibration and per-stratum decision-threshold selection.

The risk-control layer is the only *fitted* component in the model, and it acts
after the posterior exists. Two operations live here. The first is an affine
correction on the logit scale indexed by site: a correction is estimated on the
development splits and transferred unchanged to the external sites, which is what
makes the transfer a statement about a correction that has never been fitted on the
site it is applied to. The second is the per-stratum decision threshold, chosen on
the internal test split by minimising the empirical boundary error and then frozen,
with a finite-sample upper bound on the site-conditional error derived under
exchangeability within the stratum.

No calibration parameter is recomputed on external data.

Ref: Methods Sec. 4.5 (risk-control component); Sec. 4.6 (affine correction on the
logit scale, and why a flexible per-site map is rejected); Algorithm 2
(site-conditional risk control and its finite-sample bound).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from ..data.schema import StageTriple, TreatmentCategory
from ..utils.config import RiskConfig


@dataclass(frozen=True)
class AffineCorrection:
    """One affine correction on the logit scale."""

    slope: float
    intercept: float

    def apply(self, logits: np.ndarray) -> np.ndarray:
        return self.slope * logits + self.intercept

    @property
    def is_identity(self) -> bool:
        return abs(self.slope - 1.0) < 1e-12 and abs(self.intercept) < 1e-12


IDENTITY = AffineCorrection(slope=1.0, intercept=0.0)


def _fit_affine(logits: np.ndarray, targets: np.ndarray) -> AffineCorrection:
    """Logistic regression of a binary target on the single logit.

    A two-parameter fit on the logit scale, which the manuscript requires: a
    flexible per-site map would absorb exactly the between-site variation the study
    sets out to measure.
    """
    if logits.size == 0 or len(np.unique(targets)) < 2:
        return IDENTITY
    design = np.column_stack([logits, np.ones_like(logits)])
    weights = np.zeros(2, dtype=np.float64)
    for _ in range(200):
        margin = design @ weights
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(margin, -30.0, 30.0)))
        gradient = design.T @ (probabilities - targets) / len(targets)
        curvature = (design * (probabilities * (1.0 - probabilities))[:, None]).T @ design / len(targets)
        curvature += 1e-8 * np.eye(2)
        step = np.linalg.solve(curvature, gradient)
        weights -= step
        if float(np.abs(step).max()) < 1e-9:
            break
    slope = float(weights[0])
    if abs(slope) < 1e-6:
        return IDENTITY
    return AffineCorrection(slope=slope, intercept=float(weights[1]))


@dataclass
class SiteCalibrator:
    """Site-indexed affine corrections, with the leave-one-site-out variant."""

    corrections: dict[str, AffineCorrection] = field(default_factory=dict)
    transferred: AffineCorrection = IDENTITY
    leave_one_out: dict[str, AffineCorrection] = field(default_factory=dict)
    external_fitted: dict[str, AffineCorrection] = field(default_factory=dict)
    scope: str = "site"

    def correction_for(self, site: str) -> AffineCorrection:
        if self.scope == "global":
            return self.transferred
        return self.corrections.get(site, self.transferred)

    def apply(self, logits: np.ndarray, sites: list[str]) -> np.ndarray:
        """Correct each record with its own site's correction."""
        out = np.empty_like(logits, dtype=np.float64)
        for row, site in enumerate(sites):
            out[row] = self.correction_for(site).apply(np.asarray(logits[row], dtype=np.float64))
        return out

    @classmethod
    def fit(
        cls,
        logits: np.ndarray,
        targets: np.ndarray,
        sites: list[str],
        scope: str = "site",
        external_sites: tuple[str, ...] = (),
    ) -> SiteCalibrator:
        """Fit per-site corrections plus the transferred and leave-one-out variants.

        ``corrections`` are fitted on each development site's own records and are
        used for the internal layers. ``transferred`` is fitted on all development
        sites pooled and is what an external site receives, since the external sites
        contribute nothing to the fit. ``leave_one_out`` and ``external_fitted``
        exist so the manuscript's claim about a correction that is never re-fitted
        at the evaluation site can be measured against its alternative.
        """
        site_array = np.asarray(sites)
        development = np.array([site not in set(external_sites) for site in site_array])
        corrections: dict[str, AffineCorrection] = {}
        for site in sorted(set(site_array[development])):
            selector = site_array == site
            corrections[site] = _fit_affine(logits[selector], targets[selector])
        pool = development
        transferred = _fit_affine(logits[pool], targets[pool])
        leave_one_out: dict[str, AffineCorrection] = {}
        for site in sorted(set(site_array[development])):
            selector = pool & (site_array != site)
            leave_one_out[site] = _fit_affine(logits[selector], targets[selector])
        external_fitted: dict[str, AffineCorrection] = {}
        for site in sorted(set(site_array[~development])):
            selector = site_array == site
            external_fitted[site] = _fit_affine(logits[selector], targets[selector])
        return cls(
            corrections=corrections,
            transferred=transferred,
            leave_one_out=leave_one_out,
            external_fitted=external_fitted,
            scope=scope,
        )

    def as_dict(self) -> dict[str, object]:
        """Serialisable view, used in the run metadata."""
        return {
            "scope": self.scope,
            "corrections": {site: {"slope": c.slope, "intercept": c.intercept} for site, c in sorted(self.corrections.items())},
            "transferred": {"slope": self.transferred.slope, "intercept": self.transferred.intercept},
        }


@dataclass(frozen=True)
class ThresholdSelection:
    """One stratum's frozen threshold together with its finite-sample bound."""

    site: str
    threshold: float
    empirical_error: float
    bound: float
    stratum_size: int


def category_masses(projected: np.ndarray) -> np.ndarray:
    """Mass of each treatment category under a projected joint distribution.

    Returns an ``(n, 3)`` array ordered surgery-first, perioperative, systemic.
    """
    from ..data.schema import ALL_TRIPLES
    from ..data.staging import treatment_category

    order: dict[TreatmentCategory, int] = {
        TreatmentCategory.SURGERY_FIRST: 0,
        TreatmentCategory.PERIOPERATIVE: 1,
        TreatmentCategory.SYSTEMIC: 2,
    }
    mapping = np.array([order[treatment_category(stage)] for stage in ALL_TRIPLES], dtype=np.int64)
    masses = np.zeros((projected.shape[0], 3), dtype=np.float64)
    for category in range(3):
        masses[:, category] = projected[:, mapping == category].sum(axis=1)
    return masses


def decide_stage_category(masses: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Decide a management category from category masses and per-site thresholds.

    Distant or peritoneal disease is decided first against the site threshold,
    because that boundary is the one the risk layer controls; if it is not crossed,
    the remaining two categories are decided by their own masses.
    """
    systemic = masses[:, 2] >= thresholds
    earlier = np.where(masses[:, 0] >= masses[:, 1], 0, 1)
    return np.where(systemic, 2, earlier).astype(np.int64)


def select_thresholds(
    masses: np.ndarray,
    reference_categories: np.ndarray,
    sites: list[str],
    config: RiskConfig,
) -> dict[str, ThresholdSelection]:
    """Per-stratum threshold by empirical boundary-error minimisation.

    The grid is searched exhaustively, and the finite-sample bound adds the
    union-bound Hoeffding term so that the reported bound covers the whole grid
    rather than the single minimiser that happened to be selected.
    """
    site_array = np.asarray(sites)
    grid = np.asarray(config.threshold_grid, dtype=np.float64)
    if grid.size == 0:
        raise ValueError("threshold grid is empty")
    selections: dict[str, ThresholdSelection] = {}
    for site in sorted(set(sites)):
        selector = site_array == site
        stratum = int(selector.sum())
        if stratum < config.min_stratum_size:
            pooled = ThresholdSelection(site=site, threshold=0.5, empirical_error=float("nan"), bound=float("nan"), stratum_size=stratum)
            selections[site] = pooled
            continue
        block = masses[selector]
        truth = reference_categories[selector]
        errors = np.empty(grid.size, dtype=np.float64)
        for index, threshold in enumerate(grid):
            decided = decide_stage_category(block, np.full(block.shape[0], threshold))
            errors[index] = float((decided != truth).mean())
        best = int(np.argmin(errors))
        empirical = float(errors[best])
        union_term = float(np.sqrt(np.log(2.0 * grid.size / config.confidence) / (2.0 * stratum)))
        selections[site] = ThresholdSelection(
            site=site,
            threshold=float(grid[best]),
            empirical_error=empirical,
            bound=min(1.0, empirical + union_term),
            stratum_size=stratum,
        )
    return selections


def apply_thresholds(masses: np.ndarray, sites: list[str], selections: Mapping[str, ThresholdSelection], default: float = 0.5) -> np.ndarray:
    """Decide categories using the frozen per-site thresholds."""
    thresholds = np.array([selections[site].threshold if site in selections else default for site in sites], dtype=np.float64)
    return decide_stage_category(masses, thresholds)


def reference_categories(stages: list[StageTriple]) -> np.ndarray:
    """Ordinal category index of each reference stage."""
    from ..data.staging import treatment_category

    order = {
        TreatmentCategory.SURGERY_FIRST: 0,
        TreatmentCategory.PERIOPERATIVE: 1,
        TreatmentCategory.SYSTEMIC: 2,
    }
    return np.array([order[treatment_category(stage)] for stage in stages], dtype=np.int64)
