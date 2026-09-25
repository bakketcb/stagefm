"""Checkpoints.

A checkpoint carries the model state, the optimiser state, the schedule position,
the seed state, the arm's configuration and the frozen per-site thresholds, so a
resumed run continues rather than restarts. Writing is atomic: the payload lands in a
temporary file in the destination directory and is moved into place with
``os.replace``, and the temporary file is chmod-ed to 0644 first because the
``tempfile`` default would leave the checkpoint readable only by its owner.

The digest of a checkpoint hashes the tensor payloads rather than the container
bytes, because ``torch.save`` embeds storage metadata and a file hash would change
on every write even when the tensors are identical.

Ref: Methods Sec. 4.7 (thresholds frozen at the internal test split and reused
unchanged).
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ..utils.hashing import payload_digest
from ..utils.logging import get_logger
from ..utils.seeding import SeedState

LOGGER = get_logger("training.checkpoint")


@dataclass
class CheckpointState:
    """The payload of one checkpoint."""

    epoch: int
    global_step: int
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any] | None = None
    scheduler_state: dict[str, Any] | None = None
    seed_state: dict[str, Any] | None = None
    thresholds: dict[str, float] = field(default_factory=dict)
    calibration: dict[str, dict[str, float]] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    arm: str = "stagefm"


def atomic_save(payload: dict[str, Any], path: str | Path) -> Path:
    """Write a ``torch.save`` payload atomically."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    torch.save(payload, tmp)
    tmp.chmod(0o644)
    tmp.replace(target)
    return target


def save_checkpoint(state: CheckpointState, path: str | Path) -> Path:
    """Serialise one checkpoint and return the path written."""
    payload = {
        "format_version": 1,
        "arm": state.arm,
        "epoch": state.epoch,
        "global_step": state.global_step,
        "model_state": state.model_state,
        "optimizer_state": state.optimizer_state,
        "scheduler_state": state.scheduler_state,
        "seed_state": state.seed_state,
        "thresholds": state.thresholds,
        "calibration": state.calibration,
        "metrics": state.metrics,
        "config": state.config,
    }
    written = atomic_save(payload, path)
    LOGGER.info("wrote checkpoint %s", written)
    return written


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Read a checkpoint payload."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise ValueError(f"{path} is not a checkpoint written by this project")
    return payload


def restore_into(model: nn.Module, payload: dict[str, Any], optimizer: torch.optim.Optimizer | None = None, scheduler: Any | None = None) -> CheckpointState:
    """Load a checkpoint into a live model, optimiser and scheduler."""
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state") is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    return CheckpointState(
        epoch=int(payload.get("epoch", 0)),
        global_step=int(payload.get("global_step", 0)),
        model_state=dict(payload["model_state"]),
        optimizer_state=payload.get("optimizer_state"),
        scheduler_state=payload.get("scheduler_state"),
        seed_state=payload.get("seed_state"),
        thresholds=dict(payload.get("thresholds", {})),
        calibration=dict(payload.get("calibration", {})),
        metrics=dict(payload.get("metrics", {})),
        config=dict(payload.get("config", {})),
        arm=str(payload.get("arm", "stagefm")),
    )


def seed_payload(state: SeedState) -> dict[str, Any]:
    """Serialisable seed state for the checkpoint."""
    return {"seed": state.seed, "numpy": state.numpy_bit_generator_state}


def checkpoint_digest(state: CheckpointState) -> str:
    """Content digest of the tensors in a checkpoint."""
    tensors = [value for value in state.model_state.values() if isinstance(value, torch.Tensor)]
    return payload_digest(tensors)


def latest_checkpoint(directory: str | Path) -> Path | None:
    """Most recently modified checkpoint in a directory, or ``None``."""
    root = Path(directory)
    if not root.is_dir():
        return None
    candidates = sorted(root.glob("*.pt"), key=lambda item: item.stat().st_mtime)
    return candidates[-1] if candidates else None
