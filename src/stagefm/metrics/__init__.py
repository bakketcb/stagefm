"""Evaluation metrics for the staging model.

The metrics are grouped the way the manuscript reports them: discrimination on the
three axes, calibration, the treatment-boundary decision, feasibility, and the
joint-stage agreement that the pre-specified success criterion is stated on.
"""

from __future__ import annotations

from .calibration import CalibrationSummary, calibration_slope, calibration_summary, expected_calibration_error
from .concordance import axis_agreement, exact_combination_concordance, ordinal_distance_agreement, weighted_kappa
from .decision import (
    DiscordanceSummary,
    binary_nri,
    boundary_discordance,
    decision_curve,
    discordance_indicator,
    discordance_summary,
    net_benefit,
    net_reclassification_improvement,
)
from .discrimination import (
    AxisDiscrimination,
    axis_discrimination,
    binary_auroc,
    expected_axis_score,
    macro_auroc,
    ordinal_auroc,
    ordinal_auroc_ci,
    safe_binary_auroc,
)
from .feasibility import feasibility_breakdown, feasibility_rate, infeasible_examples

__all__ = [
    "AxisDiscrimination",
    "CalibrationSummary",
    "DiscordanceSummary",
    "axis_agreement",
    "axis_discrimination",
    "binary_auroc",
    "binary_nri",
    "boundary_discordance",
    "calibration_slope",
    "calibration_summary",
    "decision_curve",
    "discordance_indicator",
    "discordance_summary",
    "exact_combination_concordance",
    "expected_axis_score",
    "expected_calibration_error",
    "feasibility_breakdown",
    "feasibility_rate",
    "infeasible_examples",
    "macro_auroc",
    "net_benefit",
    "net_reclassification_improvement",
    "ordinal_auroc",
    "ordinal_auroc_ci",
    "ordinal_distance_agreement",
    "safe_binary_auroc",
    "weighted_kappa",
]
