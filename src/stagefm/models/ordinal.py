"""Monotonic ordinal heads with shared ordered cut points.

One latent score per axis passes through a set of ordered cut points, so the
axis distribution is cumulative-link by construction and the ordinal structure
cannot be disrupted by any parameter setting. The cut points are parameterised by
their positive gaps, which keeps them strictly increasing during optimisation
without a projection step.

A softmax head is also implemented, because the manuscript's justification names it
as the rejected alternative: a softmax head spends capacity on re-expressing the
ordering that the labels already carry.

Ref: Methods Sec. 4.5 (monotonic ordinal head); Sec. 4.6 (ordinal head rather than
softmax).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..utils.config import OrdinalConfig


@dataclass(frozen=True)
class OrdinalOutput:
    """Class probabilities plus the quantities the losses need."""

    probabilities: Tensor
    cumulative_logits: Tensor
    score: Tensor
    cut_points: Tensor


class MonotonicOrdinalHead(nn.Module):
    """Cumulative-link head over ``num_classes`` ordered categories."""

    def __init__(self, input_dim: int, num_classes: int, margin: float = 1e-3) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("an ordinal head needs at least two categories")
        self.num_classes = int(num_classes)
        self.margin = float(margin)
        self.score_layer = nn.Linear(input_dim, 1)
        self.raw_gaps = nn.Parameter(torch.zeros(num_classes - 1))
        nn.init.zeros_(self.score_layer.bias)
        nn.init.normal_(self.score_layer.weight, std=0.02)

    @property
    def cut_points(self) -> Tensor:
        """Strictly increasing cut points derived from the raw gaps."""
        gaps = F.softplus(self.raw_gaps) + self.margin
        return torch.cumsum(gaps, dim=0)

    def forward(self, features: Tensor) -> OrdinalOutput:
        score = self.score_layer(features).squeeze(-1)
        cuts = self.cut_points.to(score.dtype)
        cumulative = torch.sigmoid(score.unsqueeze(-1) - cuts.view(1, -1))
        ones = torch.ones_like(cumulative[..., :1])
        zeros = torch.zeros_like(cumulative[..., :1])
        upper = torch.cat([ones, cumulative], dim=-1)
        lower = torch.cat([cumulative, zeros], dim=-1)
        probabilities = torch.clamp(upper - lower, min=0.0)
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        return OrdinalOutput(
            probabilities=probabilities,
            cumulative_logits=score.unsqueeze(-1) - cuts.view(1, -1),
            score=score,
            cut_points=cuts,
        )


class SoftmaxHead(nn.Module):
    """The rejected alternative: an unordered categorical head."""

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.linear = nn.Linear(input_dim, num_classes)

    @property
    def cut_points(self) -> Tensor:
        """An unordered head has no cut points; the empty tensor says so explicitly."""
        return torch.zeros(0, device=self.linear.weight.device, dtype=self.linear.weight.dtype)

    def forward(self, features: Tensor) -> OrdinalOutput:
        logits = self.linear(features)
        probabilities = torch.softmax(logits, dim=-1)
        score = probabilities @ torch.arange(self.num_classes, device=features.device, dtype=features.dtype)
        return OrdinalOutput(
            probabilities=probabilities,
            cumulative_logits=logits,
            score=score,
            cut_points=torch.zeros(0, device=features.device, dtype=features.dtype),
        )


class StagingHeads(nn.Module):
    """The three axis heads, sharing one ordinal implementation each."""

    def __init__(self, input_dim: int, config: OrdinalConfig, ordinal: bool = True) -> None:
        super().__init__()
        margin = config.cut_point_margin
        self.ordinal = ordinal
        self.t_head: MonotonicOrdinalHead | SoftmaxHead
        self.n_head: MonotonicOrdinalHead | SoftmaxHead
        self.m_head: MonotonicOrdinalHead | SoftmaxHead
        if ordinal:
            self.t_head = MonotonicOrdinalHead(input_dim, config.t_classes, margin)
            self.n_head = MonotonicOrdinalHead(input_dim, config.n_classes, margin)
            self.m_head = MonotonicOrdinalHead(input_dim, config.m_classes, margin)
        else:
            self.t_head = SoftmaxHead(input_dim, config.t_classes)
            self.n_head = SoftmaxHead(input_dim, config.n_classes)
            self.m_head = SoftmaxHead(input_dim, config.m_classes)
        self.cardinalities = (config.t_classes, config.n_classes, config.m_classes)

    def forward(self, features: Tensor) -> dict[str, OrdinalOutput]:
        return {
            "T": self.t_head(features),
            "N": self.n_head(features),
            "M": self.m_head(features),
        }

    def cut_points(self) -> dict[str, Tensor]:
        """Cut points per axis, empty for an unordered head."""
        return {
            "T": self.t_head.cut_points,
            "N": self.n_head.cut_points,
            "M": self.m_head.cut_points,
        }

    def monotonicity_violations(self) -> int:
        """Number of cut-point comparisons that are not strictly increasing.

        Zero by construction for the ordinal head; the helper exists so the
        property is asserted in the test suite rather than assumed.
        """
        if not self.ordinal:
            return 0
        violations = 0
        for axis in ("T", "N", "M"):
            cuts = self.cut_points()[axis].detach().cpu().numpy()
            if cuts.size > 1 and not bool((cuts[1:] > cuts[:-1]).all()):
                violations += int((cuts[1:] <= cuts[:-1]).sum())
        return violations
