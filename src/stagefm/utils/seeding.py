"""Deterministic seeding.

A single entry point covers the three generators that downstream code uses:
Python's ``random``, NumPy's global state and torch's CPU and CUDA generators.
Every checkpoint stores the seed it was trained under, so a resumed run can
restore the exact stream rather than merely re-seeding.

Ref: Methods Sec. 4.7 (five independent runs per configuration).
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class SeedState:
    """The seed together with the NumPy generator state it produced."""

    seed: int
    numpy_bit_generator_state: dict[str, object]


def set_seed(seed: int) -> SeedState:
    """Seed every generator this project draws from and return the state."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    state = np.random.get_state()
    return SeedState(seed=seed, numpy_bit_generator_state={"state": state})


def restore_seed(state: SeedState) -> None:
    """Replay a previously captured seed state after a resume."""
    os.environ["PYTHONHASHSEED"] = str(state.seed)
    random.seed(state.seed)
    torch.manual_seed(state.seed)
    torch.cuda.manual_seed_all(state.seed)
    raw = state.numpy_bit_generator_state["state"]
    np.random.set_state(raw)  # type: ignore[arg-type]


def worker_init_fn(worker_id: int) -> None:
    """Give each DataLoader worker a stream that does not repeat its siblings."""
    base = torch.initial_seed() % (2**31)
    np.random.seed((base + worker_id) % (2**31))
    random.seed((base + worker_id) % (2**31))
