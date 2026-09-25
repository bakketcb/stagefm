"""Decision loss: a differentiable surrogate of the treatment-boundary error.

Treatment-boundary discordance is the share of examinations whose predicted stage
implies a different management category from the one surgical pathology supports.
The decision the layer controls is whether the patient is treated systemically, and
the projected joint gives that probability exactly, so the surrogate is the binary
cross-entropy of the systemic mass against the reference category. Its value is the
model's own estimate of the primary endpoint, which is why the constraint and the
endpoint cannot drift apart.

Ref: Algorithm 1 step 9 (second term); Methods Sec. 4.1 (endpoint definition);
Sec. 4.6 (the endpoint is primary rather than per-axis discrimination).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from ..data.schema import ALL_TRIPLES
from ..data.staging import treatment_category

CATEGORY_ORDER = {"surgery_first": 0, "perioperative": 1, "systemic": 2}


def _category_index_tensor(device: torch.device, count: int) -> Tensor:
    """Category index of each flat stage column, in the order the rule assigns."""
    order = [CATEGORY_ORDER[treatment_category(stage).value] for stage in ALL_TRIPLES[:count]]
    return torch.tensor(order, dtype=torch.long, device=device)


def systemic_mass(projected: Tensor) -> Tensor:
    """Probability that the management category is systemic, from a joint distribution."""
    mapping = _category_index_tensor(projected.device, projected.shape[-1])
    return (projected * (mapping == 2).to(projected.dtype)).sum(dim=-1)


@dataclass(frozen=True)
class DecisionLoss:
    """The surrogate endpoint value and the reference rate it is measured against."""

    total: Tensor
    systemic_probability: Tensor
    reference_systemic: Tensor


def decision_loss(projected: Tensor, reference_stages: Tensor) -> DecisionLoss:
    """Binary cross-entropy of the systemic mass against the reference category.

    The reference is passed as flat stage columns so the loss reads the same
    treatment-boundary rule the endpoint uses.
    """
    mapping = _category_index_tensor(projected.device, projected.shape[-1])
    reference_systemic = (mapping[reference_stages] == 2).to(projected.dtype)
    probability = systemic_mass(projected)
    logits = torch.log(probability.clamp(min=1e-9)) - torch.log((1.0 - probability).clamp(min=1e-9))
    total = F.binary_cross_entropy_with_logits(logits, reference_systemic)
    return DecisionLoss(total=total, systemic_probability=probability, reference_systemic=reference_systemic)


def discordance_surrogate(projected: Tensor, reference_stages: Tensor) -> Tensor:
    """Soft share of examinations whose systemic decision disagrees with the reference."""
    mapping = _category_index_tensor(projected.device, projected.shape[-1])
    reference_systemic = (mapping[reference_stages] == 2).to(projected.dtype)
    probability = systemic_mass(projected)
    return (reference_systemic * (1.0 - probability) + (1.0 - reference_systemic) * probability).mean()
