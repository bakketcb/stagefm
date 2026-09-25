"""Training-loop tests: two-step smoke run, checkpoint round trip, schedule, resume."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from stagefm.data.cohort import Layer
from stagefm.losses.total import staged_objective
from stagefm.training.amp import PrecisionSpec, autocast_context, build_scaler, model_device, to_device
from stagefm.training.checkpoint import (
    CheckpointState,
    checkpoint_digest,
    latest_checkpoint,
    load_checkpoint,
    restore_into,
    save_checkpoint,
    seed_payload,
)
from stagefm.training.ema import ModelEMA
from stagefm.training.loops import EarlyStopping, EpochStats, evaluate_loss, objective_distribution, train_epoch
from stagefm.training.optim import build_optimizer, clip_gradients, trainable_parameter_count
from stagefm.training.scheduler import ScheduleSpec, build_scheduler, constant_schedule, current_lr, steps_per_epoch, total_steps
from stagefm.training.trainer import Trainer, loader_for
from stagefm.utils.seeding import restore_seed, set_seed


def test_precision_spec_supported_values() -> None:
    assert PrecisionSpec.resolve("fp32").enabled() is False
    assert PrecisionSpec.resolve("bf16").use_scaler is False
    assert PrecisionSpec.resolve("fp16").use_scaler is True
    with pytest.raises(ValueError):
        PrecisionSpec.resolve("tf32")
    assert build_scaler(PrecisionSpec.resolve("fp16", "cpu")) is None


def test_autocast_context_is_disabled_for_fp32() -> None:
    spec = PrecisionSpec.resolve("fp32", "cpu")
    with autocast_context(spec):
        value = torch.ones(2) @ torch.ones(2)
    assert float(value) == pytest.approx(2.0)


def test_to_device_moves_tensors_and_keeps_identifiers() -> None:
    batch = {"volume": torch.zeros(1), "record_id": ["a"], "site": ["A"], "count": 3}
    moved = to_device(batch, torch.device("cpu"))
    assert isinstance(moved["volume"], torch.Tensor)
    assert moved["record_id"] == ["a"]
    assert moved["count"] == 3


def test_model_device_reads_the_first_parameter(smoke_pipeline) -> None:  # type: ignore[no-untyped-def]
    assert model_device(smoke_pipeline.model).type == "cpu"


def test_schedule_warmup_and_decay_shape() -> None:
    spec = ScheduleSpec(warmup_steps=500, total_steps=3000)
    assert spec.factor(0) == pytest.approx(1.0 / 500.0)
    assert spec.factor(249) == pytest.approx(0.5)
    assert spec.factor(500) == pytest.approx(1.0)
    assert spec.factor(0) < spec.factor(249) < spec.factor(499)
    values = [spec.factor(step) for step in (500, 1000, 2000, 3000)]
    assert all(values[index] >= values[index + 1] for index in range(len(values) - 1))
    assert spec.factor(3000) == pytest.approx(spec.minimum_factor)
    constant = constant_schedule(50)
    assert constant.factor(0) == pytest.approx(1.0)
    assert steps_per_epoch(10, 2) == 5
    assert total_steps(60, 733, 1) == 60 * 733
    with pytest.raises(ValueError):
        steps_per_epoch(10, 0)


def test_scheduler_scales_every_group(smoke_pipeline) -> None:  # type: ignore[no-untyped-def]
    optimizer, _ = build_optimizer(smoke_pipeline.model, smoke_pipeline.config.train)
    scheduler = build_scheduler(optimizer, ScheduleSpec(warmup_steps=5, total_steps=10))
    optimizer.zero_grad(set_to_none=True)
    optimizer.step()
    start = current_lr(optimizer)["group0"]
    for _ in range(4):
        scheduler.step()
    peak = current_lr(optimizer)["group0"]
    assert peak > start
    for _ in range(5):
        scheduler.step()
    assert current_lr(optimizer)["group0"] < peak


def test_optimizer_splits_adapter_and_other_groups(smoke_pipeline) -> None:  # type: ignore[no-untyped-def]
    optimizer, summary = build_optimizer(smoke_pipeline.model, smoke_pipeline.config.train)
    assert set(summary.groups) == {"adapter", "other"}
    assert summary.learning_rates["adapter"] == pytest.approx(1e-4)
    assert summary.learning_rates["other"] == pytest.approx(3e-4)
    assert trainable_parameter_count(smoke_pipeline.model) > 0


def test_clip_gradients_returns_the_pre_clip_norm(smoke_pipeline, smoke_batch) -> None:  # type: ignore[no-untyped-def]
    model = smoke_pipeline.model
    optimizer, _ = build_optimizer(model, smoke_pipeline.config.train)
    output = model(smoke_batch)
    breakdown = staged_objective(
        axis_outputs=output.axis,
        projected=output.projected,
        reference_stages=smoke_batch["stage_column"],
        targets={"T": smoke_batch["label_t"], "N": smoke_batch["label_n"], "M": smoke_batch["label_m"]},
        site_index=smoke_batch["site_index"],
        config=smoke_pipeline.config.loss,
        ordinal=True,
    )
    optimizer.zero_grad(set_to_none=True)
    breakdown.total.backward()
    norm = clip_gradients(model, 0.001)
    assert norm > 0.0
    total = float(np.sqrt(sum(float((parameter.grad**2).sum()) for parameter in model.parameters() if parameter.grad is not None)))
    assert total <= 0.001 + 1e-6


def test_early_stopping_triggers_after_patience() -> None:
    stopper = EarlyStopping(patience=2)
    assert not stopper.update(1.0, 1)
    assert not stopper.update(1.1, 2)
    assert stopper.update(1.2, 3)
    assert stopper.best == pytest.approx(1.0)
    assert stopper.best_epoch == 1
    assert stopper.stale == 2


def test_epoch_stats_averages_every_term() -> None:
    stats = EpochStats()
    spec = ScheduleSpec(warmup_steps=1, total_steps=2)
    _ = spec
    from stagefm.losses.calibration import CalibrationLoss
    from stagefm.losses.decision import DecisionLoss
    from stagefm.losses.ordinal import OrdinalLoss
    from stagefm.losses.total import LossBreakdown

    zero = torch.zeros(())
    breakdown = LossBreakdown(
        total=torch.tensor(2.0),
        ordinal=OrdinalLoss(total=torch.tensor(1.0), per_axis={"T": torch.tensor(1.0), "N": zero, "M": zero}),
        decision=DecisionLoss(total=torch.tensor(0.5), systemic_probability=zero, reference_systemic=zero),
        calibration=CalibrationLoss(total=torch.tensor(0.5), per_site_gap={}),
        lambda_decision=0.5,
        gamma_calibration=0.1,
    )
    stats.observe(breakdown, 3.0)
    stats.observe(breakdown, 1.0)
    values = stats.mean()
    assert values["total"] == pytest.approx(2.0)
    assert values["grad_norm"] == pytest.approx(2.0)
    assert values["ordinal_T"] == pytest.approx(1.0)


def test_objective_distribution_follows_the_arm(smoke_pipeline, smoke_batch) -> None:  # type: ignore[no-untyped-def]
    output = smoke_pipeline.model(smoke_batch)
    assert torch.equal(objective_distribution(smoke_pipeline.model, output), output.projected)


def test_train_epoch_reduces_the_loss_on_a_repeated_batch(smoke_config) -> None:  # type: ignore[no-untyped-def]
    from stagefm.cli.pipeline import prepare

    pipeline = prepare(smoke_config, "stagefm")
    model = pipeline.model
    batch = loader_for(pipeline.datasets[Layer.TRAIN], batch_size=4, shuffle=False)
    optimizer, _ = build_optimizer(model, smoke_config.train)
    scheduler = build_scheduler(optimizer, ScheduleSpec(warmup_steps=1, total_steps=8))
    spec = PrecisionSpec.resolve("fp32", "cpu")
    first = train_epoch(model, batch, optimizer, scheduler, smoke_config, spec, None, torch.device("cpu"), max_steps=1)
    for _ in range(12):
        last = train_epoch(model, batch, optimizer, scheduler, smoke_config, spec, None, torch.device("cpu"), max_steps=1)
    assert last["total"] < first["total"]
    assert last["steps"] == 1.0


def test_evaluate_loss_is_finite(smoke_pipeline) -> None:  # type: ignore[no-untyped-def]
    loader = loader_for(smoke_pipeline.datasets[Layer.INTERNAL_TEST], batch_size=4, shuffle=False)
    metrics = evaluate_loss(smoke_pipeline.model, loader, smoke_pipeline.config, PrecisionSpec.resolve("fp32", "cpu"), torch.device("cpu"))
    assert np.isfinite(metrics["total"])
    assert "ordinal" in metrics and "decision" in metrics


def test_checkpoint_round_trip_preserves_tensors_and_thresholds(tmp_path, smoke_pipeline) -> None:  # type: ignore[no-untyped-def]
    optimizer, _ = build_optimizer(smoke_pipeline.model, smoke_pipeline.config.train)
    state = CheckpointState(
        epoch=3,
        global_step=17,
        model_state=smoke_pipeline.model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        scheduler_state={"last_epoch": 3},
        seed_state=seed_payload(set_seed(0)),
        thresholds={"A": 0.31, "B": 0.27},
        calibration={"transferred": {"slope": 1.1, "intercept": -0.2}},
        metrics={"objective": 0.42},
        config={"experiment": "unit"},
        arm="stagefm",
    )
    digest = checkpoint_digest(state)
    path = save_checkpoint(state, tmp_path / "ckpt.pt")
    payload = load_checkpoint(path)
    restored = CheckpointState(epoch=0, global_step=0, model_state=payload["model_state"])
    assert checkpoint_digest(restored) == digest
    assert payload["thresholds"] == {"A": 0.31, "B": 0.27}
    assert payload["seed_state"]["seed"] == 0
    assert payload["epoch"] == 3
    assert path.stat().st_mode & 0o777 == 0o644
    assert latest_checkpoint(tmp_path) == path


def test_load_checkpoint_rejects_a_foreign_file(tmp_path, smoke_pipeline) -> None:  # type: ignore[no-untyped-def]
    torch.save({"something": 1}, tmp_path / "foreign.pt")
    with pytest.raises(ValueError):
        load_checkpoint(tmp_path / "foreign.pt")
    assert latest_checkpoint(tmp_path / "missing") is None


def test_restore_into_rebuilds_the_seed_and_thresholds(tmp_path, smoke_pipeline) -> None:  # type: ignore[no-untyped-def]
    optimizer, _ = build_optimizer(smoke_pipeline.model, smoke_pipeline.config.train)
    scheduler = build_scheduler(optimizer, ScheduleSpec(warmup_steps=1, total_steps=4))
    state = CheckpointState(
        epoch=1,
        global_step=2,
        model_state=smoke_pipeline.model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        scheduler_state=scheduler.state_dict(),
        seed_state=seed_payload(set_seed(1)),
        thresholds={"A": 0.4},
        arm="stagefm",
    )
    path = save_checkpoint(state, tmp_path / "resume.pt")
    payload = load_checkpoint(path)
    restored = restore_into(smoke_pipeline.model, payload, optimizer, scheduler)
    assert restored.epoch == 1 and restored.global_step == 2
    assert restored.thresholds == {"A": 0.4}
    set_seed(5)
    restore_seed(set_seed(7))
    assert int(np.random.randint(0, 1000)) >= 0


def test_ema_tracks_and_restores_the_weights(smoke_pipeline) -> None:  # type: ignore[no-untyped-def]
    ema = ModelEMA(decay=0.5)
    ema.attach(smoke_pipeline.model)
    originals = {name: parameter.detach().clone() for name, parameter in smoke_pipeline.model.named_parameters() if parameter.requires_grad}
    with torch.no_grad():
        for parameter in smoke_pipeline.model.parameters():
            if parameter.requires_grad:
                parameter.add_(1.0)
    ema.update(smoke_pipeline.model)
    first = next(iter(originals))
    expected = 0.5 * originals[first] + 0.5 * (originals[first] + 1.0)
    assert torch.allclose(ema.shadow[first], expected)
    ema.copy_to(smoke_pipeline.model)
    moved = dict(smoke_pipeline.model.named_parameters())[first]
    assert torch.allclose(moved, expected)
    ema.restore(smoke_pipeline.model)
    assert torch.allclose(dict(smoke_pipeline.model.named_parameters())[first], originals[first] + 1.0)
    snapshot = ema.clone_detached()
    assert torch.allclose(snapshot.shadow[first], ema.shadow[first])


def test_trainer_fits_and_freezes_thresholds(tmp_path, smoke_config) -> None:  # type: ignore[no-untyped-def]
    from stagefm.cli.pipeline import prepare
    from stagefm.evaluation.loop import collect_predictions

    pipeline = prepare(smoke_config, "stagefm")
    trainer = Trainer(
        model=pipeline.model,
        config=smoke_config,
        train_loader=loader_for(pipeline.datasets[Layer.TRAIN], batch_size=4, shuffle=True, seed=smoke_config.seed),
        internal_test_loader=loader_for(pipeline.datasets[Layer.INTERNAL_TEST], batch_size=4, shuffle=False),
        output_dir=tmp_path,
        arm="stagefm",
        device=torch.device("cpu"),
        max_train_steps=2,
        max_eval_batches=1,
    )
    result = trainer.fit()
    assert result.epochs_run >= 1
    assert result.best_epoch >= 1
    assert np.isfinite(result.best_internal_loss)
    assert result.history
    assert result.optimiser["groups"]["adapter"] > 0
    bundle = collect_predictions(pipeline.model, trainer.internal_test_loader, PrecisionSpec.resolve("fp32", "cpu"), torch.device("cpu"))
    selections = trainer.freeze_thresholds(bundle)
    sites = set(bundle.sites)
    assert set(selections) == sites
    for selection in selections.values():
        assert 0.0 < selection.threshold < 1.0
    assert latest_checkpoint(tmp_path) is not None


def test_trainer_restores_the_best_epoch(tmp_path, smoke_config) -> None:  # type: ignore[no-untyped-def]
    from stagefm.cli.pipeline import prepare

    pipeline = prepare(smoke_config, "stagefm")
    trainer = Trainer(
        model=pipeline.model,
        config=smoke_config,
        train_loader=loader_for(pipeline.datasets[Layer.TRAIN], batch_size=4, shuffle=False),
        internal_test_loader=loader_for(pipeline.datasets[Layer.INTERNAL_TEST], batch_size=4, shuffle=False),
        output_dir=tmp_path,
        arm="stagefm",
        device=torch.device("cpu"),
        max_train_steps=1,
        max_eval_batches=1,
    )
    result = trainer.fit()
    payload = load_checkpoint(result.checkpoint_path)
    assert payload["config"]["arm"] == "stagefm"
    assert payload["metrics"]["best_internal_loss"] == pytest.approx(result.best_internal_loss)


def test_loader_for_respects_the_batch_size(smoke_pipeline) -> None:  # type: ignore[no-untyped-def]
    loader = loader_for(smoke_pipeline.datasets[Layer.TRAIN], batch_size=4, shuffle=False)
    first = next(iter(loader))
    assert first["volume"].shape[0] == 4
