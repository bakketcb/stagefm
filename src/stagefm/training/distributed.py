"""Distributed execution.

The reported run uses four accelerators on one node. The context here reads the
standard launcher environment so the same entry point runs single-process and under
``torchrun``; only parameter averaging and the effective batch size change, never the
model definition.

Ref: Methods Sec. 4.7 (one node with four accelerators; batches of eight
examinations).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn


@dataclass(frozen=True)
class DistributedContext:
    """Rank, world size and whether the process group is active."""

    rank: int
    local_rank: int
    world_size: int
    is_distributed: bool
    backend: str
    device: torch.device

    @classmethod
    def initialise(cls, requested_world_size: int | None = None, force_cpu: bool = False) -> DistributedContext:
        """Join the launcher's process group when one exists, otherwise run alone."""
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", str(requested_world_size or 1)))
        use_cuda = torch.cuda.is_available() and not force_cpu
        backend = "nccl" if use_cuda else "gloo"
        is_distributed = world_size > 1
        if is_distributed and not dist.is_initialized():
            dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        device = torch.device(f"cuda:{local_rank}") if use_cuda else torch.device("cpu")
        if use_cuda:
            torch.cuda.set_device(device)
        return cls(rank=rank, local_rank=local_rank, world_size=world_size, is_distributed=is_distributed, backend=backend, device=device)

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def effective_batch_size(self, batch_size: int, grad_accum: int) -> int:
        """Batch size seen by one optimiser step across all ranks."""
        return batch_size * grad_accum * self.world_size

    def wrap(self, model: nn.Module) -> nn.Module:
        """Wrap in DDP when the group is active."""
        if not self.is_distributed:
            return model
        if self.device.type == "cuda":
            model = model.to(self.device)
        return nn.parallel.DistributedDataParallel(model, device_ids=[self.local_rank] if self.device.type == "cuda" else None)


def reduce_mean(value: float, context: DistributedContext) -> float:
    """Average a scalar across ranks."""
    if not context.is_distributed:
        return value
    tensor = torch.tensor([value], dtype=torch.float64, device=context.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item() / context.world_size)


def barrier(context: DistributedContext) -> None:
    """Synchronise ranks when the group is active."""
    if context.is_distributed and dist.is_initialized():
        dist.barrier()


def shutdown(context: DistributedContext) -> None:
    """Destroy the process group if this process created one."""
    if context.is_distributed and dist.is_initialized():
        dist.destroy_process_group()
