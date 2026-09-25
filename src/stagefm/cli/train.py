"""Training entry point.

Usage:

    python -m stagefm.cli.train experiment=main
    python -m stagefm.cli.train experiment=main arm=finetuned_independent_heads

The run writes its metadata, its frozen per-site thresholds and a plain-text summary
into the output directory. Thresholds are frozen at the end of training on the
internal test split, before any external data is read.

Ref: Algorithm 1; Methods Sec. 4.7.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
from typing import cast

import torch

from ..data.cohort import Layer
from ..evaluation.loop import collect_predictions
from ..training.amp import PrecisionSpec, model_device
from ..training.trainer import Trainer
from ..utils.config import ExperimentConfig, resolve_experiment
from ..utils.hashing import payload_digest
from ..utils.io import write_json, write_text
from ..utils.logging import get_logger
from ..utils.seeding import set_seed
from .pipeline import output_root, prepare, profile_table

LOGGER = get_logger("cli.train")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one arm of the staging model on one experiment config.")
    parser.add_argument("--config-dir", default="configs", help="directory holding the experiment configs")
    parser.add_argument("--experiment", default="main", help="experiment name under configs/experiment")
    parser.add_argument("--arm", default="stagefm", help="arm identifier")
    parser.add_argument("--output-dir", default=None, help="override the run output directory")
    parser.add_argument("--max-train-steps", type=int, default=None, help="cap the optimiser steps per epoch (smoke runs)")
    parser.add_argument("--max-eval-batches", type=int, default=None, help="cap the early-stopping evaluation batches")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"], help="force the compute device")
    parser.add_argument("overrides", nargs="*", help="key=value config overrides")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run one training job and write its artefacts."""
    args = parse_args(argv)
    raw = resolve_experiment(args.experiment, args.config_dir, args.overrides)
    config = ExperimentConfig.from_mapping(raw)
    if args.output_dir:
        config = _with_output_dir(config, args.output_dir)
    arm = args.arm
    run_dir = output_root(config, arm)
    run_dir.mkdir(parents=True, exist_ok=True)

    pipeline = prepare(config, arm)
    device = _device(args.device)
    set_seed(config.seed)
    trainer = Trainer(
        model=pipeline.model,
        config=config,
        train_loader=pipeline.loader(Layer.TRAIN, shuffle=True),
        internal_test_loader=pipeline.loader(Layer.INTERNAL_TEST),
        output_dir=run_dir,
        arm=arm,
        device=device,
        max_train_steps=args.max_train_steps,
        max_eval_batches=args.max_eval_batches,
    )
    result = trainer.fit()
    resolved_device = device or model_device(pipeline.model)
    spec = PrecisionSpec.resolve(config.train.precision, device_type=resolved_device.type)
    internal_bundle = collect_predictions(pipeline.model, pipeline.loader(Layer.INTERNAL_TEST), spec, resolved_device)
    selections = trainer.freeze_thresholds(internal_bundle)
    thresholds = {site: selection.threshold for site, selection in selections.items()}
    threshold_report = {site: selection.__dict__ for site, selection in selections.items()}

    metadata = {
        "experiment": config.name,
        "arm": arm,
        "seed": config.seed,
        "effective_batch_size": config.train.effective_batch_size,
        "sites": profile_table(config),
        "achievable_rule": str(pipeline.achievable.rule.value),
        "achievable_size": len(pipeline.achievable),
        "active_streams": [stream.value for stream in pipeline.plan.active],
        "encoder": dict(getattr(pipeline.model, "encoder_metadata", {})),
        "training": result.as_dict(),
        "thresholds": thresholds,
        "threshold_report": threshold_report,
        "checkpoint_digest": _checkpoint_digest(result.checkpoint_path),
        "config": _config_snapshot(config),
    }
    write_json(run_dir / "run.json", metadata)
    write_json(run_dir / "thresholds.json", thresholds)
    write_text(run_dir / "run_summary.txt", _summary_text(metadata))
    LOGGER.info("run artefacts written to %s", run_dir)
    return 0


def _device(requested: str | None) -> torch.device | None:
    """The forced device, or ``None`` so the trainer picks the model's own."""
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda")
    return None


def _checkpoint_digest(path: str) -> str:
    from ..training.checkpoint import load_checkpoint

    if not path or not Path(path).is_file():
        return ""
    payload = load_checkpoint(path)
    tensors = [value for value in payload["model_state"].values() if hasattr(value, "detach")]
    return payload_digest(tensors)


def _config_snapshot(config: ExperimentConfig) -> dict[str, object]:
    return {
        "cohort": {
            "sites": list(config.cohort.sites),
            "train_size": config.cohort.train_size,
            "internal_test_size": config.cohort.internal_test_size,
            "external_size": config.cohort.external_size,
            "prospective_size": config.cohort.prospective_size,
            "seed_count": config.cohort.seed_count,
        },
        "train": {
            "batch_size": config.train.batch_size,
            "grad_accum": config.train.grad_accum,
            "epochs": config.train.epochs,
            "patience": config.train.early_stopping_patience,
            "lr_fusion": config.train.lr_fusion,
            "lr_adapter": config.train.lr_adapter,
            "weight_decay": config.train.weight_decay,
            "warmup_steps": config.train.warmup_steps,
            "scheduler": config.train.scheduler,
            "grad_clip": config.train.grad_clip,
            "precision": config.train.precision,
            "world_size": config.train.world_size,
        },
        "encoder": {
            "name": config.encoder.name,
            "lora_rank": config.encoder.lora_rank,
            "lora_alpha": config.encoder.lora_alpha,
            "lora_targets": list(config.encoder.lora_targets),
        },
        "fusion": {
            "modality_dropout": config.fusion.modality_dropout,
            "strategy": str(config.evaluation.get("fusion_strategy", "cross_attention")),
        },
        "loss": {
            "lambda_decision": config.loss.lambda_decision,
            "gamma_calibration": config.loss.gamma_calibration,
        },
    }


def _summary_text(metadata: dict[str, object]) -> str:
    training = metadata["training"]
    assert isinstance(training, dict)
    lines = [
        "training run summary",
        f"experiment      : {metadata['experiment']}",
        f"arm             : {metadata['arm']}",
        f"seed            : {metadata['seed']}",
        f"effective batch : {metadata['effective_batch_size']}",
        f"encoder         : {metadata['encoder']}",
        f"active streams  : {', '.join(cast(Iterable[str], metadata['active_streams']))}",
        f"achievable set  : {metadata['achievable_rule']} ({metadata['achievable_size']} of 32 combinations)",
        f"best epoch      : {training['best_epoch']}",
        f"best internal   : {training['best_internal_loss']:.6f}",
        f"epochs run      : {training['epochs_run']}",
        f"steps           : {training['steps']}",
        f"checkpoint      : {training['checkpoint_path']}",
        f"elapsed seconds : {training['elapsed_seconds']:.1f}",
        f"thresholds      : {metadata['thresholds']}",
    ]
    return "\n".join(lines)


def _with_output_dir(config: ExperimentConfig, output_dir: str) -> ExperimentConfig:
    from dataclasses import replace

    evaluation = dict(config.evaluation)
    evaluation["output_dir"] = output_dir
    return replace(config, evaluation=evaluation)


if __name__ == "__main__":
    raise SystemExit(main())
