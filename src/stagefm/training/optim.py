"""Optimiser construction.

AdamW with decoupled weight decay, and two parameter groups so that the low-rank
adapters can carry their own learning rate. The weight decay applies to both groups
because the manuscript reports a single decoupled value for the whole model.

Ref: Methods Sec. 4.7 (AdamW, decoupled weight decay 1e-4, maximum learning rate
3e-4 for fusion and prediction parameters and 1e-4 for the adapters).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..models.lora import AdapterGroups
from ..utils.config import TrainConfig
from ..utils.logging import get_logger

LOGGER = get_logger("training.optim")


@dataclass(frozen=True)
class OptimiserSummary:
    """Group names and sizes, recorded in the run metadata."""

    groups: dict[str, int]
    learning_rates: dict[str, float]
    weight_decay: float

    def as_dict(self) -> dict[str, object]:
        return {"groups": self.groups, "learning_rates": self.learning_rates, "weight_decay": self.weight_decay}


def build_optimizer(model: nn.Module, config: TrainConfig) -> tuple[torch.optim.AdamW, OptimiserSummary]:
    """AdamW over the model's trainable parameters, split by adapter membership."""
    groups = AdapterGroups(model)
    parameters = groups.as_groups(lr_adapter=config.lr_adapter, lr_other=config.lr_fusion, weight_decay=config.weight_decay)
    if not parameters:
        raise ValueError("model has no trainable parameters; check the freeze settings")
    optimizer = torch.optim.AdamW(parameters, betas=(0.9, 0.999), eps=1e-8)
    summary = OptimiserSummary(
        groups={str(group["name"]): len(group["params"]) for group in parameters},  # type: ignore[arg-type]
        learning_rates={str(group["name"]): float(group["lr"]) for group in parameters},  # type: ignore[arg-type]
        weight_decay=config.weight_decay,
    )
    LOGGER.info("optimiser groups: %s", summary.as_dict())
    return optimizer, summary


def trainable_parameter_count(model: nn.Module) -> int:
    """Number of tensors the optimiser will update."""
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def clip_gradients(model: nn.Module, max_norm: float) -> float:
    """Clip the global gradient norm and return the pre-clip norm."""
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm)
    return float(norm)
