"""The trainer.

Selection runs on the internal test split and the external sites are visited once,
after the thresholds have been frozen. That order is what makes the external
evaluation a transport measurement rather than a tuning measurement, so it is
enforced by the trainer's structure: the training loop only ever sees the training
loader and the internal test loader, and threshold freezing happens before any
external loader is touched.

Ref: Methods Sec. 4.7 (selection on the internal test split, external sites visited
once, five independent runs); Algorithm 1 steps 2-13.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..models.risk_control import ThresholdSelection, reference_categories, select_thresholds
from ..utils.config import ExperimentConfig
from ..utils.logging import get_logger
from ..utils.seeding import SeedState, set_seed
from .amp import PrecisionSpec, build_scaler, model_device
from .checkpoint import CheckpointState, save_checkpoint, seed_payload
from .ema import ModelEMA
from .loops import EarlyStopping, evaluate_loss, train_epoch
from .optim import build_optimizer, trainable_parameter_count
from .scheduler import ScheduleSpec, build_scheduler, total_steps

LOGGER = get_logger("training.trainer")


@dataclass
class TrainingResult:
    """Outcome of one run of one arm."""

    arm: str
    best_epoch: int
    best_internal_loss: float
    epochs_run: int
    history: list[dict[str, float]] = field(default_factory=list)
    checkpoint_path: str = ""
    thresholds: dict[str, float] = field(default_factory=dict)
    threshold_report: dict[str, dict[str, float]] = field(default_factory=dict)
    optimiser: dict[str, Any] = field(default_factory=dict)
    parameter_counts: dict[str, float] = field(default_factory=dict)
    adapter: dict[str, float] = field(default_factory=dict)
    encoder: dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    steps: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "best_epoch": self.best_epoch,
            "best_internal_loss": self.best_internal_loss,
            "epochs_run": self.epochs_run,
            "checkpoint_path": self.checkpoint_path,
            "thresholds": self.thresholds,
            "threshold_report": self.threshold_report,
            "optimiser": self.optimiser,
            "parameter_counts": self.parameter_counts,
            "adapter": self.adapter,
            "encoder": self.encoder,
            "elapsed_seconds": self.elapsed_seconds,
            "steps": self.steps,
        }


class Trainer:
    """Fit one arm on one cohort split configuration."""

    def __init__(
        self,
        model: nn.Module,
        config: ExperimentConfig,
        train_loader: DataLoader[dict[str, Any]],
        internal_test_loader: DataLoader[dict[str, Any]],
        output_dir: str | Path,
        arm: str = "stagefm",
        seed_state: SeedState | None = None,
        device: torch.device | None = None,
        max_train_steps: int | None = None,
        max_eval_batches: int | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.internal_test_loader = internal_test_loader
        self.output_dir = Path(output_dir)
        self.arm = arm
        self.seed_state = seed_state or set_seed(config.seed)
        self.device = device or model_device(model)
        self.max_train_steps = max_train_steps
        self.max_eval_batches = max_eval_batches
        self.spec = PrecisionSpec.resolve(config.train.precision, device_type=self.device.type)
        self.scaler = build_scaler(self.spec)
        self.optimizer, self.optimiser_summary = build_optimizer(model, config.train)
        batches = min(len(train_loader), max_train_steps) if max_train_steps else len(train_loader)
        self.schedule = ScheduleSpec(
            warmup_steps=config.train.warmup_steps,
            total_steps=total_steps(config.train.epochs, batches, config.train.grad_accum),
        )
        self.scheduler = build_scheduler(self.optimizer, self.schedule)
        self.ema = ModelEMA(decay=config.train.ema_decay) if config.train.use_ema else None
        self.history: list[dict[str, float]] = []
        self.steps = 0

    def fit(self) -> TrainingResult:
        """Train with early stopping and return the training outcome."""
        started = time.perf_counter()
        stopper = EarlyStopping(self.config.train.early_stopping_patience)
        epochs_run = 0
        best_state: dict[str, Any] | None = None
        for epoch in range(1, self.config.train.epochs + 1):
            train_metrics = train_epoch(
                self.model,
                self.train_loader,
                self.optimizer,
                self.scheduler,
                self.config,
                self.spec,
                self.scaler,
                self.device,
                ema=self.ema,
                max_steps=self.max_train_steps,
            )
            self.steps += int(train_metrics.get("steps", 0.0))
            record = {f"train/{key}": value for key, value in train_metrics.items()}
            if epoch % self.config.train.eval_every == 0 or epoch == self.config.train.epochs:
                internal = evaluate_loss(self.model, self._limited(self.internal_test_loader), self.config, self.spec, self.device)
                record.update({f"internal/{key}": value for key, value in internal.items()})
                stop = stopper.update(internal["total"], epoch)
            else:
                internal = {"total": stopper.best}
                stop = False
            record["epoch"] = float(epoch)
            record["lr"] = float(self.scheduler.get_last_lr()[0])
            self.history.append(record)
            epochs_run = epoch
            if stopper.best_epoch == epoch:
                best_state = {key: value.detach().clone() for key, value in self.model.state_dict().items()}
            if epoch % self.config.train.checkpoint_every == 0:
                self._write(epoch, stopper.best, self.optimizer.state_dict())
            if stop:
                LOGGER.info("early stopping at epoch %d (best %d)", epoch, stopper.best_epoch)
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        checkpoint = self._write(epochs_run, stopper.best, self.optimizer.state_dict())
        return TrainingResult(
            arm=self.arm,
            best_epoch=stopper.best_epoch,
            best_internal_loss=stopper.best,
            epochs_run=epochs_run,
            history=self.history,
            checkpoint_path=str(checkpoint),
            optimiser=self.optimiser_summary.as_dict(),
            parameter_counts=getattr(self.model, "components", lambda: {})(),
            adapter=getattr(self.model, "adapter_report", lambda: {})(),
            encoder=dict(getattr(self.model, "encoder_metadata", {})),
            elapsed_seconds=time.perf_counter() - started,
            steps=self.steps,
        )

    def freeze_thresholds(self, bundle: Any) -> dict[str, ThresholdSelection]:
        """Choose and freeze the per-site thresholds on the internal test split.

        Runs before any external data is read, which is the order the manuscript's
        protocol requires.
        """
        from ..models.risk_control import category_masses

        masses = category_masses(bundle.projected)
        truth = reference_categories(_stages_of(bundle))
        return select_thresholds(masses, truth, list(bundle.sites), self.config.risk)

    def _limited(self, loader: DataLoader[dict[str, Any]]) -> Any:
        if self.max_eval_batches is None:
            return loader
        return _LimitedLoader(loader, self.max_eval_batches)

    def _write(self, epoch: int, best: float, optimizer_state: dict[str, Any]) -> Path:
        state = CheckpointState(
            epoch=epoch,
            global_step=self.steps,
            model_state=self.model.state_dict(),
            optimizer_state=optimizer_state,
            scheduler_state=self.scheduler.state_dict(),
            seed_state=seed_payload(self.seed_state),
            metrics={"best_internal_loss": best},
            config={"arm": self.arm, "experiment": self.config.name, "seed": self.config.seed},
            arm=self.arm,
        )
        return save_checkpoint(state, self.output_dir / f"checkpoint_epoch{epoch:04d}.pt")


def _stages_of(bundle: Any) -> list[Any]:
    from ..data.schema import ALL_TRIPLES

    return [ALL_TRIPLES[int(column)] for column in bundle.stage_column]


class _LimitedLoader:
    """A loader view that yields at most ``limit`` batches, for smoke runs only."""

    def __init__(self, loader: DataLoader[dict[str, Any]], limit: int) -> None:
        self._loader = loader
        self._limit = int(limit)

    def __iter__(self) -> Any:
        for position, batch in enumerate(self._loader):
            if position >= self._limit:
                break
            yield batch

    def __len__(self) -> int:
        return min(len(self._loader), self._limit)


def loader_for(
    dataset: Any,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    seed: int = 0,
    collate_fn: Any | None = None,
) -> DataLoader[dict[str, Any]]:
    """Build a DataLoader with the project's collate function."""
    from ..data.dataset import collate

    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn or collate,
        generator=generator if shuffle else None,
        drop_last=False,
    )


def count_trainable(model: nn.Module) -> int:
    """Re-exported for the CLI's run metadata."""
    return trainable_parameter_count(model)


def seed_summary(state: SeedState) -> dict[str, Any]:
    """Small serialisable view of the seed state."""
    return {"seed": state.seed}
