"""Low-rank adaptation of the frozen encoder's attention projections.

Adaptation is restricted to the attention projections and the rest of the encoder
stays frozen, because the target cohort is two orders of magnitude smaller than the
pre-training corpus. The adapter is zero-initialised on the output side, so the
adapted encoder starts out numerically identical to the frozen one -- which is what
makes the "frozen encoder, no adaptation" ablation a clean comparison rather than a
different initialisation.

Ref: Methods Sec. 4.5 (encoder and low-rank adaptation); Sec. 4.6 (why adaptation
rather than full fine-tuning); Sec. 4.7 (rank 16, alpha 32, targets).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class LoRALinear(nn.Module):
    """A frozen linear projection with a trainable rank-``r`` update."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.lora_a = nn.Parameter(torch.zeros(self.rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, self.rank))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Kaiming-uniform on the input side and zeros on the output side."""
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    @property
    def in_features(self) -> int:
        return int(self.base.in_features)

    @property
    def out_features(self) -> int:
        return int(self.base.out_features)

    def adapter_state(self) -> tuple[Tensor, Tensor]:
        """The two adapter matrices, for checkpoint hashing and export."""
        return self.lora_a.detach(), self.lora_b.detach()

    def forward(self, inputs: Tensor) -> Tensor:
        frozen: Tensor = self.base(inputs)
        update: Tensor = (inputs @ self.lora_a.t()) @ self.lora_b.t()
        return frozen + update * self.scaling


def _target_name(module_name: str, targets: tuple[str, ...]) -> str | None:
    for target in targets:
        if module_name == target or module_name.endswith(f".{target}"):
            return target
    return None


def apply_lora(module: nn.Module, rank: int, alpha: float, targets: tuple[str, ...]) -> int:
    """Wrap every targeted :class:`torch.nn.Linear` in place; return the count.

    Targets are matched by the leaf name of the projection, so the same call binds
    correctly to a Hugging Face style encoder (``q_proj``/``k_proj``/``v_proj``/
    ``out_proj``) and to the compact backbone shipped here.
    """
    replaced = 0
    for name, child in list(module.named_modules()):
        if not isinstance(child, nn.Linear) or isinstance(child, LoRALinear):
            continue
        if _target_name(name, targets) is None:
            continue
        parent_name, _, attribute = name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        setattr(parent, attribute, LoRALinear(child, rank=rank, alpha=alpha))
        replaced += 1
    if replaced == 0:
        raise ValueError(f"no projections matched LoRA targets {targets}")
    return replaced


def is_adapter_parameter(name: str) -> bool:
    """Whether a qualified parameter name belongs to a low-rank adapter."""
    return name.endswith("lora_a") or name.endswith("lora_b")


def freeze_except(module: nn.Module, trainable_markers: tuple[str, ...]) -> None:
    """Freeze every parameter whose qualified name contains none of ``trainable_markers``.

    The match is a substring test rather than a prefix test because a nested encoder
    qualifies its parameters (``blocks.0.attention.q_proj.lora_a``), so a prefix would
    never match.
    """
    for name, parameter in module.named_parameters():
        parameter.requires_grad_(any(marker in name for marker in trainable_markers))


def adapter_parameter_count(module: nn.Module) -> int:
    """Trainable parameters that belong to low-rank adapters."""
    return int(sum(parameter.numel() for name, parameter in module.named_parameters() if is_adapter_parameter(name) and parameter.requires_grad))


def parameter_counts(module: nn.Module) -> dict[str, float]:
    """Adapter count and share of the encoder's total parameter budget."""
    total = int(sum(parameter.numel() for parameter in module.parameters()))
    adapter = adapter_parameter_count(module)
    return {
        "encoder_total": float(total),
        "adapter_total": float(adapter),
        "adapter_fraction": (adapter / total) if total else 0.0,
    }


class AdapterGroups:
    """Named parameter groups so the optimiser can give adapters their own learning rate."""

    def __init__(self, module: nn.Module) -> None:
        self.adapter: list[nn.Parameter] = []
        self.other: list[nn.Parameter] = []
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue
            if is_adapter_parameter(name):
                self.adapter.append(parameter)
            else:
                self.other.append(parameter)

    def as_groups(self, lr_adapter: float, lr_other: float, weight_decay: float) -> list[dict[str, object]]:
        """Optimiser group list, omitting empty groups."""
        groups: list[dict[str, object]] = []
        if self.adapter:
            groups.append({"params": self.adapter, "lr": lr_adapter, "weight_decay": weight_decay, "name": "adapter"})
        if self.other:
            groups.append({"params": self.other, "lr": lr_other, "weight_decay": weight_decay, "name": "other"})
        return groups
