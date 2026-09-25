"""The training objective: ordinal + decision + calibration.

The three terms are combined with the weights the config supplies, so the arm's
``lambda`` and ``gamma`` are visible in the config rather than buried in the loss.

Ref: Algorithm 1 step 9.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from ..models.ordinal import OrdinalOutput
from ..utils.config import LossConfig
from .calibration import CalibrationLoss, calibration_loss
from .decision import DecisionLoss, decision_loss
from .ordinal import OrdinalLoss, ordinal_loss


@dataclass(frozen=True)
class LossBreakdown:
    """Every term of the objective plus the weighted total."""

    total: Tensor
    ordinal: OrdinalLoss
    decision: DecisionLoss
    calibration: CalibrationLoss
    lambda_decision: float
    gamma_calibration: float

    def as_floats(self) -> dict[str, float]:
        """Scalar view for logging."""
        values = {
            "loss/total": float(self.total.detach()),
            "loss/ordinal": float(self.ordinal.total.detach()),
            "loss/decision": float(self.decision.total.detach()),
            "loss/calibration": float(self.calibration.total.detach()),
        }
        for axis, term in self.ordinal.per_axis.items():
            values[f"loss/ordinal_{axis.lower()}"] = float(term.detach())
        return values


def staged_objective(
    axis_outputs: dict[str, OrdinalOutput],
    projected: Tensor,
    reference_stages: Tensor,
    targets: dict[str, Tensor],
    site_index: Tensor,
    config: LossConfig,
    ordinal: bool = True,
) -> LossBreakdown:
    """Combine the three terms for one batch."""
    ordinal_term = ordinal_loss(axis_outputs, targets, ordinal=ordinal)
    decision_term = decision_loss(projected, reference_stages)
    calibration_term = calibration_loss(projected, reference_stages, site_index, variant=config.calibration_variant)
    total = ordinal_term.total + config.lambda_decision * decision_term.total + config.gamma_calibration * calibration_term.total
    return LossBreakdown(
        total=total,
        ordinal=ordinal_term,
        decision=decision_term,
        calibration=calibration_term,
        lambda_decision=config.lambda_decision,
        gamma_calibration=config.gamma_calibration,
    )
