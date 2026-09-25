"""Training and evaluation loops.

One step is: forward, objective, backward, gradient clip, optimiser step, schedule
step. Gradient accumulation is handled by dividing each microbatch's loss by the
accumulation count, so the effective batch reported in the config is the batch the
optimiser actually sees.

The objective is computed on the projected distribution whenever the arm enforces
feasibility, and on the raw joint for the unconstrained arm whose projection happens
after the fact -- which is what makes that arm a control for the projection rather
than a control for the loss.

Ref: Algorithm 1 steps 3-11.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

from ..losses.total import LossBreakdown, staged_objective
from ..models.stagefm import StageOutput
from ..utils.config import ExperimentConfig
from ..utils.logging import get_logger
from .amp import PrecisionSpec, autocast_context, to_device
from .optim import clip_gradients
from .scheduler import current_lr

LOGGER = get_logger("training.loops")


@dataclass
class EpochStats:
    """Aggregated loss terms and gradient norms over one epoch."""

    steps: int = 0
    total: float = 0.0
    ordinal: float = 0.0
    decision: float = 0.0
    calibration: float = 0.0
    grad_norm: float = 0.0
    per_axis: dict[str, float] = field(default_factory=dict)

    def observe(self, breakdown: LossBreakdown, grad_norm: float) -> None:
        self.steps += 1
        self.total += float(breakdown.total.detach())
        self.ordinal += float(breakdown.ordinal.total.detach())
        self.decision += float(breakdown.decision.total.detach())
        self.calibration += float(breakdown.calibration.total.detach())
        self.grad_norm += grad_norm
        for axis, term in breakdown.ordinal.per_axis.items():
            self.per_axis[axis] = self.per_axis.get(axis, 0.0) + float(term.detach())

    def mean(self) -> dict[str, float]:
        divisor = max(self.steps, 1)
        values = {
            "total": self.total / divisor,
            "ordinal": self.ordinal / divisor,
            "decision": self.decision / divisor,
            "calibration": self.calibration / divisor,
            "grad_norm": self.grad_norm / divisor,
        }
        for axis, value in self.per_axis.items():
            values[f"ordinal_{axis}"] = value / divisor
        return values


def objective_distribution(model: nn.Module, output: StageOutput) -> Tensor:
    """The distribution the objective is computed on for this arm."""
    flags = getattr(model, "flags", None)
    if flags is not None and not getattr(flags, "stage_consistency", True):
        chosen: Tensor = output.joint
    else:
        chosen = output.projected
    return chosen


def is_ordinal(model: nn.Module) -> bool:
    """Whether the arm's heads are cumulative-link rather than softmax."""
    flags = getattr(model, "flags", None)
    if flags is None:
        return True
    return bool(getattr(flags, "ordinal", True))


def compute_loss(model: nn.Module, output: StageOutput, batch: dict[str, Any], config: ExperimentConfig) -> LossBreakdown:
    """Assemble the objective for one batch."""
    targets = {"T": batch["label_t"], "N": batch["label_n"], "M": batch["label_m"]}
    distribution = objective_distribution(model, output)
    return staged_objective(
        axis_outputs=output.axis,
        projected=distribution,
        reference_stages=batch["stage_column"],
        targets=targets,
        site_index=batch["site_index"],
        config=config.loss,
        ordinal=is_ordinal(model),
    )


def train_epoch(
    model: nn.Module,
    loader: Iterable[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    config: ExperimentConfig,
    spec: PrecisionSpec,
    scaler: Any | None,
    device: torch.device,
    ema: Any | None = None,
    max_steps: int | None = None,
) -> dict[str, float]:
    """Run one epoch and return the mean loss terms."""
    model.train()
    stats = EpochStats()
    accum = max(config.train.grad_accum, 1)
    optimizer.zero_grad(set_to_none=True)
    for position, raw_batch in enumerate(loader):
        if max_steps is not None and position >= max_steps:
            break
        batch = to_device(raw_batch, device)
        with autocast_context(spec):
            output = model(batch)
            breakdown = compute_loss(model, output, batch, config)
            loss = breakdown.total / accum
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()  # type: ignore[no-untyped-call]
        if (position + 1) % accum == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            norm = clip_gradients(model, config.train.grad_clip)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            if ema is not None:
                ema.update(model)
            stats.observe(breakdown, norm)
    result = stats.mean()
    result["steps"] = float(stats.steps)
    result.update(current_lr(optimizer))
    return result


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: Iterable[dict[str, Any]],
    config: ExperimentConfig,
    spec: PrecisionSpec,
    device: torch.device,
) -> dict[str, float]:
    """Mean objective terms over a loader, used for early stopping."""
    model.eval()
    stats = EpochStats()
    for raw_batch in loader:
        batch = to_device(raw_batch, device)
        with autocast_context(spec):
            output = model(batch)
            breakdown = compute_loss(model, output, batch, config)
        stats.observe(breakdown, 0.0)
    return stats.mean()


class EarlyStopping:
    """Patience-based stopping on a monitored value, minimised."""

    def __init__(self, patience: int, minimum_delta: float = 1e-6) -> None:
        self.patience = int(patience)
        self.minimum_delta = float(minimum_delta)
        self.best = float("inf")
        self.best_epoch = 0
        self.stale = 0

    def update(self, value: float, epoch: int) -> bool:
        """Record a value; return True when training should stop."""
        if value < self.best - self.minimum_delta:
            self.best = value
            self.best_epoch = epoch
            self.stale = 0
            return False
        self.stale += 1
        return self.stale >= self.patience
