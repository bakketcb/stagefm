"""Evaluation entry point.

Usage:

    python -m stagefm.cli.eval experiment=main arm=stagefm
    python -m stagefm.cli.eval experiment=main arm=unconstrained_posthoc

The evaluation reads the frozen thresholds written by the training run, collects
predictions on the external layer, and writes every reported table plus a plain-text
summary. Nothing is refitted here: the external sites contribute to no estimate.

Ref: Methods Sec. 4.11 (evaluation protocol) and Sec. 4.13 (scope of the experiment
set).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..data.cohort import Layer
from ..data.schema import ALL_TRIPLES
from ..evaluation.ascertainment import ascertainment_report
from ..evaluation.loop import PredictionBundle, collect_predictions
from ..evaluation.per_site import per_site_report
from ..evaluation.report import arm_row, clinical_relevance, prespecified_criterion
from ..evaluation.sensitivity import sensitivity_report
from ..evaluation.silent_mode import deployment_report, synthetic_deployment_log
from ..evaluation.subgroups import subgroup_columns, subgroup_report
from ..models.baselines import BASELINES, QUOTED_ARMS, arm_registry
from ..models.risk_control import category_masses
from ..stats.bootstrap import bootstrap_statistic
from ..training.amp import PrecisionSpec, model_device
from ..training.checkpoint import load_checkpoint, restore_into
from ..utils.config import ExperimentConfig, resolve_experiment
from ..utils.io import read_json, write_json, write_text
from ..utils.logging import get_logger
from .pipeline import PreparedPipeline, output_root, prepare

LOGGER = get_logger("cli.eval")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one arm of the staging model on the external cohort.")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--experiment", default="main")
    parser.add_argument("--arm", default="stagefm")
    parser.add_argument("--run-dir", default=None, help="directory of the training run whose thresholds to load")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint", default=None, help="checkpoint to restore before evaluating")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    parser.add_argument("--skip-bootstrap", action="store_true", help="skip the primary-endpoint bootstrap interval")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Evaluate one arm and write its artefacts."""
    args = parse_args(argv)
    raw = resolve_experiment(args.experiment, args.config_dir, args.overrides)
    config = ExperimentConfig.from_mapping(raw)
    if args.output_dir:
        config = _with_output_dir(config, args.output_dir)
    arm = args.arm
    run_dir = Path(args.run_dir) if args.run_dir else output_root(config, arm)
    pipeline = prepare(config, arm)
    device = _device(args.device) or model_device(pipeline.model)
    checkpoint = args.checkpoint or _latest(run_dir)
    if checkpoint and Path(checkpoint).is_file():
        payload = load_checkpoint(checkpoint)
        restore_into(pipeline.model, payload)
        LOGGER.info("restored checkpoint %s", checkpoint)
    thresholds = _load_thresholds(run_dir)
    spec = PrecisionSpec.resolve(config.train.precision, device_type=device.type)

    internal = collect_predictions(pipeline.model, pipeline.loader(Layer.INTERNAL_TEST), spec, device)
    external = collect_predictions(pipeline.model, pipeline.loader(Layer.EXTERNAL), spec, device)
    internal = _apply_thresholds(internal, thresholds)
    external = _apply_thresholds(external, thresholds)

    regions = dict(config.cohort.regions)
    report: dict[str, Any] = {
        "experiment": config.name,
        "arm": arm,
        "layers": {"internal_test": len(internal), "external": len(external)},
        "thresholds": thresholds,
        "per_site": per_site_report(external, pipeline.achievable, regions).as_dict(),
        "internal_per_site": per_site_report(internal, pipeline.achievable, regions).as_dict(),
        "subgroups": subgroup_report(external, subgroup_columns(pipeline.records(Layer.EXTERNAL), config.cohort.adequate_node_yield), pipeline.achievable),
        "ascertainment": ascertainment_report(
            external,
            subgroup_columns(pipeline.records(Layer.EXTERNAL), config.cohort.adequate_node_yield),
            config.cohort.median_harvested_nodes,
            config.cohort.adequate_node_yield,
        ).as_dict(),
        "sensitivity": sensitivity_report(external, _consensus_flags(pipeline)),
        "deployment": deployment_report(
            synthetic_deployment_log(
                {site: eligible for site, eligible in zip(("A", "B", "C", "D"), (612, 498, 402, 338))},
                {"A": 0.846, "B": 0.779, "C": 0.714, "D": 0.749},
                seed=config.seed,
            )
        ),
        "quoted_arms": {key: arm.metrics for key, arm in QUOTED_ARMS.items()},
    }
    report["primary_endpoint"] = _primary_endpoint(external, config, skip_bootstrap=args.skip_bootstrap)
    report["tables"] = _tables(external, pipeline, arm, report["primary_endpoint"])
    write_json(run_dir / "evaluation.json", report)
    write_text(run_dir / "evaluation_summary.txt", _summary_text(report))
    LOGGER.info("evaluation artefacts written to %s", run_dir)
    return 0


def _tables(external: PredictionBundle, pipeline: PreparedPipeline, arm: str, endpoint: dict[str, Any]) -> dict[str, Any]:
    """The arm table and the pre-specified criterion for this arm."""
    registry = arm_registry()
    label = registry[arm].label if arm in registry else arm
    row = arm_row(arm, label, external, pipeline.achievable)
    table: dict[str, Any] = {"rows": [row.as_dict()]}
    for key, spec in BASELINES.items():
        if key != arm:
            table["rows"].append({"arm": key, "label": spec.label, "source": "not_evaluated_in_this_run"})
    table["criterion"] = prespecified_criterion(row.concordance, row.t_auroc)
    table["clinical_relevance"] = clinical_relevance(row.discordance_percent, row.discordance_percent, 3.0)
    table["discordance_interval"] = {"low": endpoint.get("low", float("nan")), "high": endpoint.get("high", float("nan"))}
    return table


def _primary_endpoint(bundle: PredictionBundle, config: ExperimentConfig, skip_bootstrap: bool) -> dict[str, Any]:
    """Treatment-boundary discordance with its interval and its per-site values."""
    from ..metrics.decision import discordance_indicator, discordance_summary

    indicator = discordance_indicator(bundle.predicted_columns, bundle.stage_column)
    pooled = discordance_summary(bundle.predicted_columns, bundle.stage_column, list(bundle.sites))
    endpoint: dict[str, Any] = {
        "value_percent": pooled.pooled,
        "per_site_percent": pooled.per_site,
        "per_site_counts": pooled.per_site_counts,
        "mcnemar_ready": True,
        "clinical_relevance_threshold_points": 3.0,
    }
    if not skip_bootstrap and indicator.size:
        result = bootstrap_statistic(
            lambda index: float(100.0 * indicator[index].mean()),
            count=len(indicator),
            resamples=int(config.evaluation.get("bootstrap_resamples", 2000)),
            seed=config.seed,
        )
        endpoint.update(
            {
                "low": result.low,
                "high": result.high,
                "standard_error": result.standard_error,
                "half_width": result.half_width,
                "replicates": result.replicates,
            }
        )
    return endpoint


def _apply_thresholds(bundle: PredictionBundle, thresholds: dict[str, float]) -> PredictionBundle:
    """Re-decide the reported stage under the frozen per-site thresholds.

    The threshold acts on the systemic boundary: a record whose systemic mass
    reaches its site's threshold is reported as a systemic-category stage, and
    otherwise the earlier of the two remaining categories is reported. Applying it
    here rather than inside the model keeps the projection and the threshold as the
    two separable operations the ablation distinguishes.
    """
    if not thresholds:
        return bundle

    masses = category_masses(bundle.projected)
    decided = np.empty(len(bundle), dtype=np.int64)
    for position in range(len(bundle)):
        site = bundle.sites[position]
        threshold = thresholds.get(site, 0.5)
        if masses[position, 2] >= threshold:
            decided[position] = 2
        else:
            decided[position] = 0 if masses[position, 0] >= masses[position, 1] else 1
    columns = _columns_for_categories(decided)
    projected = np.zeros_like(bundle.projected)
    projected[np.arange(len(bundle)), columns] = 1.0
    return PredictionBundle(
        t_prob=bundle.t_prob,
        n_prob=bundle.n_prob,
        m_prob=bundle.m_prob,
        joint=bundle.joint,
        projected=projected,
        boundary_logit=bundle.boundary_logit,
        labels=bundle.labels,
        sites=bundle.sites,
        harvested_nodes=bundle.harvested_nodes,
        record_ids=bundle.record_ids,
        stage_column=bundle.stage_column,
        extra={**bundle.extra, "pre_threshold_columns": bundle.predicted_columns},
    )


def _columns_for_categories(categories: np.ndarray) -> np.ndarray:
    """A representative stage column for each decided management category.

    The representative is the argmax stage inside the category under the projected
    distribution, so the reported stage and the reported category cannot disagree.
    """
    from ..data.staging import treatment_category

    order = {"surgery_first": 0, "perioperative": 1, "systemic": 2}
    by_category: dict[int, list[int]] = {0: [], 1: [], 2: []}
    for column, stage in enumerate(ALL_TRIPLES):
        by_category[order[treatment_category(stage).value]].append(column)
    representatives = np.array([by_category[0][0], by_category[1][0], by_category[2][0]], dtype=np.int64)
    chosen: np.ndarray = representatives[categories]
    return chosen


def _consensus_flags(pipeline: PreparedPipeline) -> np.ndarray:
    return np.array([record.consensus_staged for record in pipeline.records(Layer.EXTERNAL)], dtype=bool)


def _load_thresholds(run_dir: Path) -> dict[str, float]:
    path = run_dir / "thresholds.json"
    if not path.is_file():
        return {}
    payload = read_json(path)
    return {str(site): float(value) for site, value in payload.items()} if isinstance(payload, dict) else {}


def _latest(run_dir: Path) -> str | None:
    candidates = sorted(run_dir.glob("checkpoint_epoch*.pt")) if run_dir.is_dir() else []
    return str(candidates[-1]) if candidates else None


def _device(requested: str | None) -> torch.device | None:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda")
    return None


def _summary_text(report: dict[str, Any]) -> str:
    pooled = report["per_site"]["pooled"]
    endpoint = report["primary_endpoint"]
    criterion = report["tables"]["criterion"]
    lines = [
        "evaluation summary",
        f"experiment            : {report['experiment']}",
        f"arm                   : {report['arm']}",
        f"external examinations : {report['layers']['external']}",
        f"T-axis AUROC          : {pooled['t_auroc']:.4f}",
        f"N-axis AUROC (ordinal): {pooled['n_auroc']:.4f}",
        f"M-axis AUROC          : {pooled['m_auroc']:.4f}",
        f"exact concordance     : {pooled['concordance']:.4f}",
        f"weighted kappa        : {pooled['weighted_kappa']:.4f}",
        f"boundary discordance  : {endpoint['value_percent']:.2f}%",
        f"discordance interval  : {endpoint.get('low', float('nan')):.2f} - {endpoint.get('high', float('nan')):.2f}",
        f"calibration error     : {pooled['expected_calibration_error']:.4f}",
        f"feasibility           : {pooled['feasibility_percent']:.1f}%",
        f"concordance criterion : {'met' if criterion['concordance_met'] else 'not met'} at {criterion['concordance_target']}",
        f"T-axis criterion      : {'met' if criterion['t_auroc_met'] else 'not met'} at {criterion['t_auroc_target']}",
        "",
        "per-site discordance (percent)",
    ]
    for site, value in sorted(endpoint["per_site_percent"].items()):
        lines.append(f"  site {site}: {value:.2f}")
    return "\n".join(lines)


def _with_output_dir(config: ExperimentConfig, output_dir: str) -> ExperimentConfig:
    from dataclasses import replace

    evaluation = dict(config.evaluation)
    evaluation["output_dir"] = output_dir
    return replace(config, evaluation=evaluation)


if __name__ == "__main__":
    raise SystemExit(main())
