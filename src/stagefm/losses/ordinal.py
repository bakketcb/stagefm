"""Ordinal (cumulative-link) loss.

The negative log-likelihood of the proportional-odds model on the head's cumulative
logits. Writing it this way rather than as a cross-entropy on the derived class
probabilities keeps the gradient attached to the ordered cut points directly, so a
cut point that the data wants moved is moved.

Ref: Algorithm 1 step 9 (the first term of the objective); Methods Sec. 4.5
(monotonic ordinal head).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from ..models.ordinal import OrdinalOutput


@dataclass(frozen=True)
class OrdinalLoss:
    """Per-axis ordinal loss and its total."""

    total: Tensor
    per_axis: dict[str, Tensor]


def cumulative_link_loss(cumulative_logits: Tensor, targets: Tensor) -> Tensor:
    """Mean binary cross-entropy over the cumulative thresholds.

    ``cumulative_logits[b, k]`` is the logit of ``P(Y > k)`` for record ``b``; the
    target for that entry is ``1`` when the observed category exceeds ``k``.
    """
    if cumulative_logits.ndim != 2:
        raise ValueError("cumulative logits must be (batch, thresholds)")
    thresholds = cumulative_logits.shape[1]
    if thresholds == 0:
        return cumulative_logits.sum() * 0.0
    levels = torch.arange(thresholds, device=targets.device).view(1, -1)
    binary_targets = (targets.unsqueeze(-1) > levels).to(cumulative_logits.dtype)
    return F.binary_cross_entropy_with_logits(cumulative_logits, binary_targets)


def class_probability_loss(probabilities: Tensor, targets: Tensor) -> Tensor:
    """Ordinary cross-entropy on the class probabilities, for the unordered head."""
    return F.nll_loss(torch.log(probabilities.clamp(min=1e-12)), targets)


def ordinal_loss(axis: dict[str, OrdinalOutput], targets: dict[str, Tensor], ordinal: bool = True) -> OrdinalLoss:
    """Sum the axis losses for one batch."""
    per_axis: dict[str, Tensor] = {}
    for name in ("T", "N", "M"):
        output = axis[name]
        if ordinal:
            per_axis[name] = cumulative_link_loss(output.cumulative_logits, targets[name])
        else:
            per_axis[name] = class_probability_loss(output.probabilities, targets[name])
    total = per_axis["T"] + per_axis["N"] + per_axis["M"]
    return OrdinalLoss(total=total, per_axis=per_axis)
