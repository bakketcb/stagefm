"""Training utilities: optimiser, schedule, precision, checkpoints, loops, EMA."""

from __future__ import annotations

from .amp import PrecisionSpec, autocast_context, build_scaler, model_device, to_device
from .checkpoint import CheckpointState, atomic_save, checkpoint_digest, latest_checkpoint, load_checkpoint, restore_into, save_checkpoint, seed_payload
from .distributed import DistributedContext, barrier, reduce_mean, shutdown
from .ema import ModelEMA
from .loops import EarlyStopping, EpochStats, compute_loss, evaluate_loss, objective_distribution, train_epoch
from .optim import OptimiserSummary, build_optimizer, clip_gradients, trainable_parameter_count
from .scheduler import ScheduleSpec, build_scheduler, constant_schedule, current_lr, steps_per_epoch, total_steps
from .trainer import Trainer, TrainingResult, count_trainable, loader_for, seed_summary

__all__ = [
    "CheckpointState",
    "DistributedContext",
    "EarlyStopping",
    "EpochStats",
    "ModelEMA",
    "OptimiserSummary",
    "PrecisionSpec",
    "ScheduleSpec",
    "Trainer",
    "TrainingResult",
    "atomic_save",
    "autocast_context",
    "barrier",
    "build_optimizer",
    "build_scaler",
    "build_scheduler",
    "checkpoint_digest",
    "clip_gradients",
    "compute_loss",
    "constant_schedule",
    "count_trainable",
    "current_lr",
    "evaluate_loss",
    "latest_checkpoint",
    "loader_for",
    "load_checkpoint",
    "model_device",
    "objective_distribution",
    "reduce_mean",
    "restore_into",
    "save_checkpoint",
    "seed_payload",
    "seed_summary",
    "shutdown",
    "steps_per_epoch",
    "to_device",
    "total_steps",
    "train_epoch",
    "trainable_parameter_count",
]
