"""In vitro lymphangiogenesis analysis.

The assay asks whether the peritumoral texture feature that carries the nodal signal
reflects an active lymphangiogenic programme rather than a passive morphological
correlation. The estimators here are the ones the protocol calls for: the
tube-formation index and the VEGFR-3 phosphorylation ratio expressed as multiples of
the matched unconditioned control with a bootstrap interval, the small-interfering-RNA
knockdown effect against a non-targeting control, a blocked analysis with the plate as
the blocking factor, and the correlation between the texture feature and lymphatic
vessel density measured on resection material.

The measurement block is not part of this release; the protocol, the design table and
the estimators are, so a real block can be dropped in without changing the analysis.

Ref: Methods Sec. 2.10 and Sec. 4.10 (in vitro protocol and its reporting).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..stats.bootstrap import percentile_interval
from ..stats.correlation import spearman_with_bootstrap

BIOLOGICAL_REPLICATES = 6
BLINDED_REMEASURE_FRACTION = 0.10
DAYS_TO_MEDIUM_COLLECTION = 2
MAX_PASSAGES = 10
LEC_PASSAGE_RANGE = (3, 8)


@dataclass(frozen=True)
class InVitroCondition:
    """One assay condition."""

    name: str
    vegfc_expression: str
    knockdown: bool = False
    control: bool = False

    def as_dict(self) -> dict[str, str | bool]:
        return {"name": self.name, "vegfc_expression": self.vegfc_expression, "knockdown": self.knockdown, "control": self.control}


def conditions() -> list[InVitroCondition]:
    """The arm set the protocol specifies."""
    return [
        InVitroCondition(name="unconditioned", vegfc_expression="none", control=True),
        InVitroCondition(name="high_vegfc", vegfc_expression="high"),
        InVitroCondition(name="low_vegfc", vegfc_expression="low"),
        InVitroCondition(name="high_vegfc_sirna", vegfc_expression="high", knockdown=True),
        InVitroCondition(name="non_targeting_control", vegfc_expression="high", knockdown=False),
        InVitroCondition(name="lymphatic_endothelial_only", vegfc_expression="none"),
    ]


def design_table(replicates: int = BIOLOGICAL_REPLICATES, plates: int = 2) -> list[dict[str, object]]:
    """Condition by replicate by plate layout, with the plate as a blocking factor."""
    rows: list[dict[str, object]] = []
    for plate in range(plates):
        for condition in conditions():
            for replicate in range(replicates):
                rows.append(
                    {
                        "plate": plate,
                        "condition": condition.name,
                        "replicate": replicate,
                        "knockdown": condition.knockdown,
                        "control": condition.control,
                    }
                )
    return rows


@dataclass(frozen=True)
class RatioEstimate:
    """A multiple-of-control estimate with its bootstrap interval."""

    condition: str
    ratio: float
    low: float
    high: float
    replicates: int

    def as_dict(self) -> dict[str, object]:
        return {"condition": self.condition, "ratio": self.ratio, "low": self.low, "high": self.high, "replicates": self.replicates}


def ratio_to_control(
    measurement: np.ndarray,
    condition: np.ndarray,
    control_condition: str,
    target_condition: str,
    resamples: int = 2000,
    seed: int = 0,
) -> RatioEstimate:
    """Target-to-control ratio with a percentile bootstrap interval.

    Each replicate's value is expressed relative to the matched control run on the
    same plate, which is what removes the plate effect from the ratio before the
    interval is drawn.
    """
    target = measurement[condition == target_condition]
    control = measurement[condition == control_condition]
    if target.size == 0 or control.size == 0 or float(control.mean()) == 0.0:
        return RatioEstimate(condition=target_condition, ratio=float("nan"), low=float("nan"), high=float("nan"), replicates=0)
    point = float(target.mean() / control.mean())
    rng = np.random.default_rng(seed)
    replicates = np.empty(resamples, dtype=np.float64)
    for position in range(resamples):
        draw_target = target[rng.integers(0, target.size, size=target.size)]
        draw_control = control[rng.integers(0, control.size, size=control.size)]
        denominator = float(draw_control.mean())
        replicates[position] = float(draw_target.mean() / denominator) if denominator != 0.0 else np.nan
    low, high = percentile_interval(replicates, 0.05)
    return RatioEstimate(condition=target_condition, ratio=point, low=low, high=high, replicates=resamples)


def blocked_condition_effect(measurement: np.ndarray, condition: np.ndarray, plate: np.ndarray) -> dict[str, float]:
    """Two-way analysis without interaction, with the plate as the blocking factor.

    Returns the condition and plate sums of squares and the residual variance, which
    is the decomposition a blocked design reports.
    """
    measurement = np.asarray(measurement, dtype=np.float64)
    condition = np.asarray(condition)
    plate = np.asarray(plate)
    grand = float(measurement.mean())
    conditions = np.unique(condition)
    plates = np.unique(plate)
    ss_condition = float(sum(len(measurement[condition == name]) * (measurement[condition == name].mean() - grand) ** 2 for name in conditions))
    ss_plate = float(sum(len(measurement[plate == name]) * (measurement[plate == name].mean() - grand) ** 2 for name in plates))
    ss_total = float(((measurement - grand) ** 2).sum())
    ss_residual = max(ss_total - ss_condition - ss_plate, 0.0)
    df_condition = len(conditions) - 1
    df_plate = len(plates) - 1
    df_residual = max(len(measurement) - len(conditions) - len(plates) + 1, 1)
    return {
        "ss_condition": ss_condition,
        "ss_plate": ss_plate,
        "ss_residual": ss_residual,
        "ms_condition": ss_condition / df_condition if df_condition > 0 else float("nan"),
        "ms_plate": ss_plate / df_plate if df_plate > 0 else float("nan"),
        "ms_residual": ss_residual / df_residual,
        "df_condition": float(df_condition),
        "df_plate": float(df_plate),
        "df_residual": float(df_residual),
    }


def knockdown_reversal(measurement: np.ndarray, condition: np.ndarray, control_condition: str, treated: str, knocked_down: str) -> dict[str, float]:
    """Effect of the knockdown on a measured response and on the control response."""
    treated_ratio = ratio_to_control(measurement, condition, control_condition, treated)
    knocked_ratio = ratio_to_control(measurement, condition, control_condition, knocked_down)
    return {
        "treated_ratio": treated_ratio.ratio,
        "treated_low": treated_ratio.low,
        "treated_high": treated_ratio.high,
        "knockdown_ratio": knocked_ratio.ratio,
        "knockdown_low": knocked_ratio.low,
        "knockdown_high": knocked_ratio.high,
    }


def texture_density_correlation(texture: np.ndarray, vessel_density: np.ndarray, resamples: int = 2000, seed: int = 0) -> dict[str, float]:
    """Spearman correlation between the nodal texture feature and lymphatic vessel density."""
    return spearman_with_bootstrap(np.asarray(texture, dtype=np.float64), np.asarray(vessel_density, dtype=np.float64), resamples=resamples, seed=seed)


def blinded_remeasurement_agreement(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    """Agreement between the two blinded technicians on the re-measured subset."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.size < 2 or first.size != second.size:
        return {"correlation": float("nan"), "mean_difference": float("nan"), "count": float(first.size)}
    difference = second - first
    return {
        "correlation": float(np.corrcoef(first, second)[0, 1]),
        "mean_difference": float(difference.mean()),
        "standard_deviation": float(difference.std(ddof=1)),
        "count": float(first.size),
    }


@dataclass
class InVitroReport:
    """Assembled in vitro report; empty when no measurement block was supplied."""

    design_rows: int = 0
    tube_formation: dict[str, float] = field(default_factory=dict)
    phosphorylation: dict[str, float] = field(default_factory=dict)
    blocked_model: dict[str, float] = field(default_factory=dict)
    texture_correlation: dict[str, float] = field(default_factory=dict)
    measured: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "design_rows": self.design_rows,
            "tube_formation": self.tube_formation,
            "phosphorylation": self.phosphorylation,
            "blocked_model": self.blocked_model,
            "texture_correlation": self.texture_correlation,
            "measured": self.measured,
        }


def in_vitro_report(
    tube_formation: np.ndarray | None = None,
    phosphorylation: np.ndarray | None = None,
    condition: np.ndarray | None = None,
    plate: np.ndarray | None = None,
    texture: np.ndarray | None = None,
    vessel_density: np.ndarray | None = None,
) -> InVitroReport:
    """Assemble the report from an optional measurement block."""
    report = InVitroReport(design_rows=len(design_table()))
    if tube_formation is None or condition is None:
        return report
    report.measured = True
    report.tube_formation = knockdown_reversal(tube_formation, condition, "unconditioned", "high_vegfc", "high_vegfc_sirna")
    if plate is not None:
        report.blocked_model = blocked_condition_effect(tube_formation, condition, plate)
    if phosphorylation is not None:
        report.phosphorylation = knockdown_reversal(phosphorylation, condition, "unconditioned", "high_vegfc", "high_vegfc_sirna")
    if texture is not None and vessel_density is not None:
        report.texture_correlation = texture_density_correlation(texture, vessel_density)
    return report
