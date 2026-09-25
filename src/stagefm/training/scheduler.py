"""Learning-rate schedule: linear warmup followed by cosine decay.

The schedule is a step-indexed function so the number of warmup steps is the
manuscript's fixed 500 rather than a fraction of the run, and the minimum factor is
kept above zero so a resumed run at the very end of the schedule still moves. The
warmup ramp starts at one over the warmup length rather than at zero, so the first
optimiser step is not spent with a learning rate of exactly zero.

Ref: Methods Sec. 4.7 (cosine decay with a linear warmup over the first 500 steps).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.optim.lr_scheduler import LambdaLR


@dataclass(frozen=True)
class ScheduleSpec:
    """Warmup length, total length and the floor factor."""

    warmup_steps: int
    total_steps: int
    minimum_factor: float = 0.01

    def factor(self, step: int) -> float:
        """Multiplier applied to each group's base learning rate at ``step``."""
        if self.total_steps <= 0:
            return 1.0
        if self.warmup_steps > 0 and step < self.warmup_steps:
            # The first optimiser step already carries a non-zero rate, so a warmup
            # step is not spent at a learning rate of exactly zero.
            return float(max(step, 0) + 1) / float(self.warmup_steps)
        if self.total_steps <= self.warmup_steps:
            return self.minimum_factor
        progress = (step - self.warmup_steps) / float(self.total_steps - self.warmup_steps)
        progress = float(min(max(progress, 0.0), 1.0))
        cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793)).item())
        return float(self.minimum_factor + (1.0 - self.minimum_factor) * cosine)


def build_scheduler(optimizer: torch.optim.Optimizer, spec: ScheduleSpec) -> LambdaLR:
    """A LambdaLR that scales every group by the schedule factor."""
    return LambdaLR(optimizer, lr_lambda=lambda step: spec.factor(step))


def constant_schedule(total_steps: int) -> ScheduleSpec:
    """The ablation schedule used where no decay is wanted."""
    return ScheduleSpec(warmup_steps=0, total_steps=max(total_steps, 1), minimum_factor=1.0)


def steps_per_epoch(batches: int, grad_accum: int) -> int:
    """Optimiser steps in one epoch."""
    if grad_accum <= 0:
        raise ValueError("gradient accumulation must be positive")
    return max(1, batches // grad_accum)


def total_steps(epochs: int, batches: int, grad_accum: int) -> int:
    """Optimiser steps across a run."""
    return max(1, epochs * steps_per_epoch(batches, grad_accum))


def current_lr(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    """Learning rate per parameter group, for logging."""
    return {f"group{index}": float(group["lr"]) for index, group in enumerate(optimizer.param_groups)}
