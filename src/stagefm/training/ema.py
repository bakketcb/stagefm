"""Exponential moving average of model weights.

Off by default, because the reported profile does not use it; when enabled it keeps a
shadow copy that is evaluated for selection and can be swapped in at export. The
buffer is excluded from the shadow so a resumed run does not carry a stale buffer.

Ref: Methods Sec. 4.7 (the reported optimisation profile).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ModelEMA:
    """A decayed copy of a model's parameters."""

    decay: float = 0.999
    shadow: dict[str, torch.Tensor] | None = None
    backup: dict[str, torch.Tensor] | None = None

    def attach(self, model: nn.Module) -> None:
        """Snapshot the current parameters as the initial shadow."""
        self.shadow = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}

    def update(self, model: nn.Module) -> None:
        """Blend the live trainable parameters into the shadow."""
        if self.shadow is None:
            self.attach(model)
            return
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad or name not in self.shadow:
                    continue
                self.shadow[name].mul_(self.decay).add_(parameter.detach(), alpha=1.0 - self.decay)

    def copy_to(self, model: nn.Module) -> None:
        """Install the shadow into a model, keeping the live values aside."""
        if self.shadow is None:
            return
        self.backup = {}
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name not in self.shadow:
                    continue
                self.backup[name] = parameter.detach().clone()
                parameter.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        """Undo :meth:`copy_to`."""
        if self.backup is None:
            return
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in self.backup:
                    parameter.copy_(self.backup[name])
        self.backup = None

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {} if self.shadow is None else {name: tensor.clone() for name, tensor in self.shadow.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.shadow = {name: tensor.clone() for name, tensor in state.items()}

    def clone_detached(self) -> ModelEMA:
        """A detached copy, used when the EMA is evaluated without disturbing the model."""
        return ModelEMA(decay=self.decay, shadow=copy.deepcopy(self.shadow) if self.shadow else None)
