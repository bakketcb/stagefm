"""Staging definitions: the achievable set A and the treatment-boundary rule.

Three objects live here and all three are fixed before any model is trained.

``AchievableSet`` describes the joint label space A that the stage-consistency
block projects onto. It is built from the category definitions themselves rather
than fitted to the cohort (Methods Sec. 4.6), and it exposes oversized and
undersized variants because the sensitivity analysis recomputes the projection with
the boundary of A moved in both directions.

``treatment_category`` maps a stage onto a management category. It reads only the
stage, so a site name can never enter the primary endpoint (Methods Sec. 4.1).

``boundary_discordant`` compares two stages through that mapping.

Ref: Methods Sec. 4.1 (treatment-boundary labels), Sec. 4.5 (stage-consistency
block), Sec. 4.6 (justification of the load-bearing choices), Sec. 4.11
(sensitivity analyses).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .schema import ALL_TRIPLES, M_CLASSES, N_CLASSES, T_CLASSES, Axis, StageTriple, TreatmentCategory


class AchievableRule(str, Enum):
    """Named constructions of the achievable set.

    ``STAGING`` is the definition shipped as the default. ``STAGING_LOOSE`` and
    ``STAGING_TIGHT`` are the oversized and undersized variants the sensitivity
    analysis moves the boundary of A to. ``DATA_DERIVED`` is kept only as the
    deliberately rejected alternative named in Methods Sec. 4.6.
    """

    STAGING = "staging"
    STAGING_LOOSE = "staging_loose"
    STAGING_TIGHT = "staging_tight"
    DATA_DERIVED = "data_derived"


def _staging_achievable(t: int, n: int, m: int) -> bool:
    """Category definitions applied to one triple.

    Two properties of the categories themselves carry the constraint. A T1 lesion
    carries at most N1 involvement and is not recorded with distant metastasis,
    because the T1 category is defined by invasion no deeper than the submucosa and
    the cohort's M category records peritoneal or distant disease that requires a
    deeper primary. A T2 lesion carries at most N2. Distant metastasis with no
    regional node involvement is only reachable when the primary has already
    extended to T4.
    """
    if t == 1 and (n > 1 or m == 1):
        return False
    if t == 2 and n > 2:
        return False
    return not (n == 0 and m == 1 and t < 4)


def _loose_achievable(t: int, n: int, m: int) -> bool:
    """Oversized A: only the node categories that the T category cannot support."""
    if t == 1 and n > 1:
        return False
    return not (t == 1 and m == 1)


def _tight_achievable(t: int, n: int, m: int) -> bool:
    """Undersized A: distant disease only with an already advanced, node-positive primary."""
    if not _staging_achievable(t, n, m):
        return False
    return m == 0 or (t == 4 and n >= 1)


_RULES = {
    AchievableRule.STAGING: _staging_achievable,
    AchievableRule.STAGING_LOOSE: _loose_achievable,
    AchievableRule.STAGING_TIGHT: _tight_achievable,
}


@dataclass(frozen=True)
class AchievableSet:
    """The joint label space A as a boolean mask over the 32 combinations."""

    rule: AchievableRule = AchievableRule.STAGING
    # Normalised in __post_init__: the default is None only so that a rule-derived
    # mask can be computed before the field is read.
    mask: np.ndarray = field(default=None)  # type: ignore[arg-type]

    def __post_init__(self) -> None:
        mask = self.mask
        if mask is None:
            if self.rule is AchievableRule.DATA_DERIVED:
                raise ValueError("DATA_DERIVED requires an explicit mask built from observed labels")
            predicate = _RULES[self.rule]
            mask = np.array([predicate(stage.t, stage.n, stage.m) for stage in ALL_TRIPLES], dtype=bool)
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (len(ALL_TRIPLES),):
            raise ValueError(f"A must be a length-{len(ALL_TRIPLES)} mask, got {mask.shape}")
        if not mask.any():
            raise ValueError("A is empty; the projection would have no feasible support")
        object.__setattr__(self, "mask", mask)

    @classmethod
    def from_rule(cls, rule: AchievableRule | str) -> AchievableSet:
        return cls(rule=AchievableRule(rule))

    @classmethod
    def from_observed(cls, triples: Iterable[StageTriple]) -> AchievableSet:
        """The rejected alternative: A fitted to the labels seen in the cohort."""
        mask = np.zeros(len(ALL_TRIPLES), dtype=bool)
        for stage in triples:
            mask[stage.flat_index] = True
        return cls(rule=AchievableRule.DATA_DERIVED, mask=mask)

    def __len__(self) -> int:
        return int(self.mask.sum())

    def contains(self, stage: StageTriple) -> bool:
        return bool(self.mask[stage.flat_index])

    @property
    def feasible_fraction(self) -> float:
        """Share of the joint label space that A admits."""
        return float(self.mask.mean())

    def triples(self) -> tuple[StageTriple, ...]:
        return tuple(stage for stage, keep in zip(ALL_TRIPLES, self.mask.tolist()) if keep)

    def exclude_from(self, triples: Iterable[StageTriple]) -> list[StageTriple]:
        return [stage for stage in triples if not self.contains(stage)]


def treatment_category(stage: StageTriple) -> TreatmentCategory:
    """Map a stage onto a management category.

    The split follows the trial evidence the manuscript cites for the preoperative
    decision (Ref. Sec. 1, Ref. [1]): a T1 lesion without nodal involvement is
    resected first, locally advanced but resectable disease receives perioperative
    chemotherapy before surgery, and distant or peritoneal disease is treated
    systemically.
    """
    if stage.m == 1:
        return TreatmentCategory.SYSTEMIC
    if stage.t == 1 and stage.n == 0:
        return TreatmentCategory.SURGERY_FIRST
    return TreatmentCategory.PERIOPERATIVE


def boundary_discordant(predicted: StageTriple, reference: StageTriple) -> bool:
    """True when the two stages imply different management categories."""
    return treatment_category(predicted) is not treatment_category(reference)


_CATEGORY_ORDER = {
    TreatmentCategory.SURGERY_FIRST: 0,
    TreatmentCategory.PERIOPERATIVE: 1,
    TreatmentCategory.SYSTEMIC: 2,
}


def shift_category(stage: StageTriple, offset: int) -> TreatmentCategory:
    """The category of ``stage`` moved by ``offset`` steps along the management order.

    Used only by the sensitivity analysis that moves the boundary rule's categories
    one step in each direction (Methods Sec. 4.11).
    """
    current = _CATEGORY_ORDER[treatment_category(stage)]
    shifted = min(max(current + offset, 0), len(_CATEGORY_ORDER) - 1)
    for category, order in _CATEGORY_ORDER.items():
        if order == shifted:
            return category
    raise AssertionError("category order is not dense")


def axis_cardinalities() -> dict[Axis, int]:
    """Number of categories on each axis."""
    return {Axis.T: T_CLASSES, Axis.N: N_CLASSES, Axis.M: M_CLASSES}


def axis_index(stage: StageTriple, axis: Axis) -> int:
    """Zero-based class index of ``stage`` on ``axis`` (the T axis is offset back to zero)."""
    if axis is Axis.T:
        return stage.t - 1
    if axis is Axis.N:
        return stage.n
    return stage.m
