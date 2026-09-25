"""Mixed-precision context.

Precision is a configuration value, not a global switch: the run reports the
precision it used and the context manager is the only place that decides. ``bf16``
needs no loss scaling, ``fp16`` does, and ``fp32`` runs without autocast.

Ref: Methods Sec. 4.7 (the training profile's precision).
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

SUPPORTED = ("fp32", "bf16", "fp16")


@dataclass(frozen=True)
class PrecisionSpec:
    """The resolved precision policy."""

    name: str
    autocast_dtype: torch.dtype
    use_scaler: bool
    device_type: str

    @classmethod
    def resolve(cls, name: str, device_type: str = "cpu") -> PrecisionSpec:
        if name not in SUPPORTED:
            raise ValueError(f"unsupported precision {name!r}; expected one of {', '.join(SUPPORTED)}")
        if name == "bf16":
            return cls(name=name, autocast_dtype=torch.bfloat16, use_scaler=False, device_type=device_type)
        if name == "fp16":
            return cls(name=name, autocast_dtype=torch.float16, use_scaler=True, device_type=device_type)
        return cls(name=name, autocast_dtype=torch.float32, use_scaler=False, device_type=device_type)

    def enabled(self) -> bool:
        return self.name != "fp32"


def autocast_context(spec: PrecisionSpec) -> AbstractContextManager[None]:
    """Autocast on the accelerator when a reduced precision was requested."""
    if not spec.enabled() or spec.device_type != "cuda":
        return torch.autocast(device_type="cpu", enabled=False)
    return torch.autocast(device_type=spec.device_type, dtype=spec.autocast_dtype)


def build_scaler(spec: PrecisionSpec) -> Any:
    """A gradient scaler for fp16, ``None`` otherwise."""
    if not spec.use_scaler or spec.device_type != "cuda":
        return None
    return torch.amp.GradScaler(spec.device_type)  # type: ignore[attr-defined]


def to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    """Move every tensor in a batch to the device, leaving identifiers alone."""
    moved: dict[str, object] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def model_device(model: nn.Module) -> torch.device:
    """Device of the first parameter, for the autocast decision."""
    for parameter in model.parameters():
        return parameter.device
    return torch.device("cpu")
