"""Results assembly.

One place turns prediction bundles into the rows the manuscript reports: the
all-axis arm table, the ablation table with its interaction decomposition, and the
pre-specified success criterion. Keeping the assembly in one module is what makes the
report and the verification pass read the same numbers.

Ref: Table 3 (all-axis arms), Table 4 (component ablation and interaction), Sec. 4.12
(pre-specified success criterion), Fig. 1 (discordance by arm).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..data.staging import AchievableSet
from ..models.baselines import QUOTED_ARMS
from ..stats.multiplicity import InteractionDecomposition
from .loop import PredictionBundle
from .per_site import pooled_summary


@dataclass(frozen=True)
class ArmRow:
    """One row of the all-axis arm table."""

    arm: str
    label: str
    source: str
    t_auroc: float
    n_auroc: float
    m_auroc: float
    concordance: float
    weighted_kappa: float
    discordance_percent: float
    expected_calibration_error: float
    feasibility_percent: float
    count: int
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "label": self.label,
            "source": self.source,
            "t_auroc": self.t_auroc,
            "n_auroc": self.n_auroc,
            "m_auroc": self.m_auroc,
            "concordance": self.concordance,
            "weighted_kappa": self.weighted_kappa,
            "discordance_percent": self.discordance_percent,
            "expected_calibration_error": self.expected_calibration_error,
            "feasibility_percent": self.feasibility_percent,
            "count": self.count,
            "detail": self.detail,
        }


def arm_row(arm: str, label: str, bundle: PredictionBundle, achievable: AchievableSet, source: str = "computed") -> ArmRow:
    """Summarise one arm's predictions into a table row."""
    summary = pooled_summary(bundle, achievable)
    return ArmRow(
        arm=arm,
        label=label,
        source=source,
        t_auroc=summary["t_auroc"],
        n_auroc=summary["n_auroc"],
        m_auroc=summary["m_auroc"],
        concordance=summary["concordance"],
        weighted_kappa=summary["weighted_kappa"],
        discordance_percent=summary["discordance_percent"],
        expected_calibration_error=summary["expected_calibration_error"],
        feasibility_percent=summary["feasibility_percent"],
        count=int(summary["count"]),
    )


def quoted_row(arm: str, label: str) -> ArmRow:
    """A reference point whose values come from the manuscript, not from this code."""
    values = QUOTED_ARMS[arm].metrics
    return ArmRow(
        arm=arm,
        label=label,
        source="quoted",
        t_auroc=float(values.get("t_auroc", float("nan"))),
        n_auroc=float(values.get("n_auroc", float("nan"))),
        m_auroc=float(values.get("m_auroc", float("nan"))),
        concordance=float(values.get("concordance", float("nan"))),
        weighted_kappa=float(values.get("weighted_kappa", float("nan"))),
        discordance_percent=float(values.get("discordance", float("nan"))),
        expected_calibration_error=float(values.get("ece", float("nan"))),
        feasibility_percent=float(values.get("feasibility", float("nan"))),
        count=int(values.get("count", 0)),
        detail={"reference": QUOTED_ARMS[arm].source, "values": dict(values)},
    )


def best_unconstrained(rows: list[ArmRow], arm_keys: tuple[str, ...] = ("finetuned_independent_heads", "unconstrained_posthoc")) -> ArmRow | None:
    """The strongest unconstrained arm on the tumour axis, for the parity comparison."""
    candidates = [row for row in rows if row.arm in arm_keys and np.isfinite(row.t_auroc)]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.t_auroc)


def parity_check(model_row: ArmRow, unconstrained: ArmRow, tolerance: float = 0.02) -> dict[str, float]:
    """Per-axis parity of the constrained model against the best unconstrained arm.

    Parity is the claim the manuscript makes on the axes: a restricted label space
    cannot buy discrimination, because the feasible set is a subset of the free one.
    """
    return {
        "t_difference": model_row.t_auroc - unconstrained.t_auroc,
        "n_difference": model_row.n_auroc - unconstrained.n_auroc,
        "m_difference": model_row.m_auroc - unconstrained.m_auroc,
        "max_abs_difference": max(
            abs(model_row.t_auroc - unconstrained.t_auroc),
            abs(model_row.n_auroc - unconstrained.n_auroc),
            abs(model_row.m_auroc - unconstrained.m_auroc),
        ),
        "within_tolerance": float(
            max(
                abs(model_row.t_auroc - unconstrained.t_auroc),
                abs(model_row.n_auroc - unconstrained.n_auroc),
                abs(model_row.m_auroc - unconstrained.m_auroc),
            )
            < tolerance
        ),
    }


def prespecified_criterion(concordance: float, t_auroc: float, concordance_target: float = 0.80, t_target: float = 0.90) -> dict[str, Any]:
    """Evaluate the two pre-specified success criteria."""
    return {
        "concordance_target": concordance_target,
        "concordance_observed": concordance,
        "concordance_met": bool(np.isfinite(concordance) and concordance >= concordance_target),
        "t_auroc_target": t_target,
        "t_auroc_observed": t_auroc,
        "t_auroc_met": bool(np.isfinite(t_auroc) and t_auroc >= t_target),
        "both_met": bool(np.isfinite(concordance) and np.isfinite(t_auroc) and concordance >= concordance_target and t_auroc >= t_target),
    }


def clinical_relevance(discordance_model: float, discordance_control: float, threshold_points: float = 3.0) -> dict[str, float]:
    """Reduction against a control arm against the pre-specified clinical threshold."""
    reduction = discordance_control - discordance_model
    return {
        "control_discordance": discordance_control,
        "model_discordance": discordance_model,
        "reduction_points": reduction,
        "threshold_points": threshold_points,
        "exceeds_threshold": float(reduction >= threshold_points),
    }


def interaction_from_rows(rows: dict[str, float], first: str, second: str, reference: str, full: str) -> InteractionDecomposition:
    """Build an interaction decomposition from four named ablation rows.

    ``without_first`` is the arm in which the first component is absent, so the second
    component's separate effect is read from it.
    """
    return InteractionDecomposition(
        reference=rows[reference],
        without_first=rows[first],
        without_second=rows[second],
        full=rows[full],
    )


@dataclass(frozen=True)
class AblationRow:
    """One row of the component ablation."""

    key: str
    label: str
    t_auroc: float
    n_auroc: float
    m_auroc: float
    concordance: float
    discordance_percent: float
    feasibility_percent: float
    non_monotonic: bool = False
    note: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "label": self.label,
            "t_auroc": self.t_auroc,
            "n_auroc": self.n_auroc,
            "m_auroc": self.m_auroc,
            "concordance": self.concordance,
            "discordance_percent": self.discordance_percent,
            "feasibility_percent": self.feasibility_percent,
            "non_monotonic": self.non_monotonic,
            "note": self.note,
        }


def ablation_row(key: str, label: str, bundle: PredictionBundle, achievable: AchievableSet, non_monotonic: bool = False, note: str = "") -> AblationRow:
    """Summarise one ablation arm."""
    summary = pooled_summary(bundle, achievable)
    return AblationRow(
        key=key,
        label=label,
        t_auroc=summary["t_auroc"],
        n_auroc=summary["n_auroc"],
        m_auroc=summary["m_auroc"],
        concordance=summary["concordance"],
        discordance_percent=summary["discordance_percent"],
        feasibility_percent=summary["feasibility_percent"],
        non_monotonic=non_monotonic,
        note=note,
    )


def interaction_section(discordance_rows: dict[str, float], concordance_rows: dict[str, float]) -> dict[str, Any]:
    """Both interaction ratios, computed from the ablation rows.

    The decision endpoint and the exact-combination concordance are both reported
    because they carry opposite verdicts: the components act super-additively on the
    decision and sub-additively on the concordance, which separates decision safety
    from label-space constraint.
    """
    decision = interaction_from_rows(
        discordance_rows,
        first="without_stage_consistency",
        second="without_risk_control",
        reference="without_both_structural",
        full="full",
    )
    concordance = interaction_from_rows(
        concordance_rows,
        first="without_stage_consistency",
        second="without_risk_control",
        reference="without_both_structural",
        full="full",
    )
    return {"discordance": decision.as_dict(), "concordance": concordance.as_dict()}


def results_table(rows: list[ArmRow], model_arm: str = "stagefm") -> dict[str, Any]:
    """Assemble the table with the parity comparison attached."""
    model = next((row for row in rows if row.arm == model_arm), None)
    strongest = best_unconstrained(rows)
    out: dict[str, Any] = {"rows": [row.as_dict() for row in rows]}
    if model is not None and strongest is not None:
        out["parity"] = {"against": strongest.arm, **parity_check(model, strongest)}
        out["criterion"] = prespecified_criterion(model.concordance, model.t_auroc)
    return out
