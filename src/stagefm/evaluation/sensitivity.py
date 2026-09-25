"""Sensitivity analyses.

Two analyses are pre-specified. The first removes the examinations whose stage came
from tumour-board consensus rather than from resection, which tests whether the
endpoint depends on records whose reference standard is weaker. The second moves the
boundary rule's categories one step in each direction, which tests whether the
result depends on where the boundaries were placed.

Both recompute the endpoint on the same predictions; neither refits anything.

Ref: Methods Sec. 4.11 (the two sensitivity analyses); Sec. 4.1 (the boundary rule).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..data.schema import ALL_TRIPLES, TreatmentCategory
from ..data.staging import shift_category
from .loop import PredictionBundle


@dataclass(frozen=True)
class SensitivityResult:
    """One sensitivity variant's endpoint values."""

    name: str
    count: int
    discordance_percent: float
    concordance: float
    detail: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "count": self.count,
            "discordance_percent": self.discordance_percent,
            "concordance": self.concordance,
            "detail": self.detail or {},
        }


def _order() -> dict[TreatmentCategory, int]:
    """Position of each management category along the decision order."""
    return {
        TreatmentCategory.SURGERY_FIRST: 0,
        TreatmentCategory.PERIOPERATIVE: 1,
        TreatmentCategory.SYSTEMIC: 2,
    }


def _category_index(column: int) -> int:
    from ..data.staging import treatment_category

    return _order()[treatment_category(ALL_TRIPLES[column])]


def _discordance_from_indices(predicted_indices: np.ndarray, reference_indices: np.ndarray) -> float:
    if predicted_indices.size == 0:
        return float("nan")
    return float(100.0 * (predicted_indices != reference_indices).mean())


def consensus_exclusion(bundle: PredictionBundle, consensus_flags: np.ndarray) -> SensitivityResult:
    """Endpoint recomputed without examinations staged by board consensus."""
    flags = np.asarray(consensus_flags, dtype=bool)
    keep = ~flags
    predicted = bundle.predicted_columns[keep]
    reference = bundle.stage_column[keep]
    predicted_indices = np.array([_category_index(int(column)) for column in predicted], dtype=np.int64)
    reference_indices = np.array([_category_index(int(column)) for column in reference], dtype=np.int64)
    concordance = float((predicted == reference).mean()) if predicted.size else float("nan")
    return SensitivityResult(
        name="exclude_consensus_staged",
        count=int(keep.sum()),
        discordance_percent=_discordance_from_indices(predicted_indices, reference_indices),
        concordance=concordance,
        detail={"excluded": int(flags.sum())},
    )


def boundary_shift(bundle: PredictionBundle, offsets: tuple[int, ...] = (-1, 0, 1)) -> list[SensitivityResult]:
    """Endpoint recomputed with both arms' categories shifted along the order.

    A zero offset reproduces the reported endpoint, so the analysis carries its own
    control: if the zero-offset value did not match the primary result the variant
    would be measuring something other than the boundary rule.
    """
    predicted = bundle.predicted_columns
    reference = bundle.stage_column
    results: list[SensitivityResult] = []
    for predicted_offset in offsets:
        for reference_offset in offsets:
            predicted_indices = np.array([_order()[shift_category(ALL_TRIPLES[int(c)], predicted_offset)] for c in predicted], dtype=np.int64)
            reference_indices = np.array([_order()[shift_category(ALL_TRIPLES[int(c)], reference_offset)] for c in reference], dtype=np.int64)
            concordance = float((predicted_indices == reference_indices).mean()) if predicted_indices.size else float("nan")
            results.append(
                SensitivityResult(
                    name=f"predicted_shift{predicted_offset:+d}_reference_shift{reference_offset:+d}",
                    count=int(predicted.size),
                    discordance_percent=_discordance_from_indices(predicted_indices, reference_indices),
                    concordance=concordance,
                    detail={"predicted_offset": predicted_offset, "reference_offset": reference_offset},
                )
            )
    return results


def sensitivity_report(bundle: PredictionBundle, consensus_flags: np.ndarray | None = None) -> dict[str, Any]:
    """Both pre-specified analyses, plus the control case of the shift analysis."""
    shifts = boundary_shift(bundle)
    control = next(result for result in shifts if result.name == "predicted_shift+0_reference_shift+0")
    report: dict[str, Any] = {
        "boundary_shift": [result.as_dict() for result in shifts],
        "control_discordance_percent": control.discordance_percent,
    }
    if consensus_flags is not None:
        report["consensus_exclusion"] = consensus_exclusion(bundle, consensus_flags).as_dict()
    return report
