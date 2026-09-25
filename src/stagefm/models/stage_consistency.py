"""Joint stage distribution and the differentiable feasibility projection.

The three axis distributions are combined into a joint distribution over nominal
stage combinations, and the joint is projected onto the achievable set A so that
every combination outside the staging definitions receives zero mass and the
remaining mass is renormalised. The projection is deterministic and differentiable,
so the constraint can sit inside the training graph rather than being applied
afterwards.

The degenerate case -- no mass on the feasible set -- returns the uniform
distribution on A rather than an undefined normalisation.

Ref: Methods Sec. 4.5 (stage-consistency block); Algorithm 1 step 8; Algorithm 3
(feasibility projection); Sec. 4.6 (A from staging definitions).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from ..data.schema import ALL_TRIPLES, M_CLASSES, N_CLASSES, T_CLASSES, StageTriple
from ..data.staging import AchievableSet


def joint_distribution(t_prob: Tensor, n_prob: Tensor, m_prob: Tensor) -> Tensor:
    """Outer product of the three axis distributions, flattened to the 32 combinations."""
    if t_prob.shape[-1] != T_CLASSES or n_prob.shape[-1] != N_CLASSES or m_prob.shape[-1] != M_CLASSES:
        raise ValueError("axis distributions do not match the declared cardinalities")
    joint = torch.einsum("bi,bj,bk->bijk", t_prob, n_prob, m_prob)
    return joint.reshape(joint.shape[0], -1)


@dataclass(frozen=True)
class ProjectionOutput:
    """Projected joint distribution and the normaliser it used."""

    probabilities: Tensor
    normaliser: Tensor
    degenerate: Tensor


class FeasibilityProjection(nn.Module):
    """Project a joint stage distribution onto the achievable set."""

    mask: Tensor

    def __init__(self, achievable: AchievableSet) -> None:
        super().__init__()
        feasible = torch.as_tensor(np.asarray(achievable.mask, dtype=bool))
        self.register_buffer("mask", feasible)
        self.rule = achievable.rule

    @property
    def feasible_count(self) -> int:
        return int(self.mask.sum().item())

    @property
    def uniform_on_feasible(self) -> Tensor:
        """The degenerate-case distribution: uniform over A."""
        weights = self.mask.to(torch.float32)
        return weights / weights.sum().clamp(min=1.0)

    def forward(self, joint: Tensor) -> ProjectionOutput:
        if joint.shape[-1] != self.mask.shape[0]:
            raise ValueError(f"joint must have {self.mask.shape[0]} entries, got {joint.shape[-1]}")
        restricted = joint * self.mask.to(joint.dtype)
        normaliser = restricted.sum(dim=-1, keepdim=True)
        degenerate = normaliser.squeeze(-1) <= 1e-12
        safe = torch.where(normaliser > 1e-12, normaliser, torch.ones_like(normaliser))
        projected = restricted / safe
        if bool(degenerate.any()):
            uniform = self.uniform_on_feasible.to(projected.dtype).view(1, -1).expand_as(projected)
            projected = torch.where(degenerate.unsqueeze(-1), uniform, projected)
        return ProjectionOutput(probabilities=projected, normaliser=normaliser.squeeze(-1), degenerate=degenerate)

    def is_feasible(self, columns: Tensor) -> Tensor:
        """Whether each flat stage column lies in A."""
        return self.mask.to(columns.device)[columns]


def column_to_stage(column: int) -> StageTriple:
    """The stage triple at a flat column index."""
    return ALL_TRIPLES[column]
