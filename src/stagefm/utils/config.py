"""Configuration loading and the typed config objects passed between modules.

The experiment YAML under ``configs/experiment`` composes four partial files
(``data/cohort``, ``model/default``, ``train/default`` and an optional per-arm
override) and accepts ``key=value`` command-line overrides. The loaded mapping is
then bound to the dataclasses below so that no module reads an untyped dictionary.

Ref: Methods Sec. 4.7 (optimisation settings) and Sec. 4.5 (module layout).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_DIR = Path("configs")
_MISSING = object()


class ConfigError(RuntimeError):
    """Raised when a config key is absent or has the wrong shape."""


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read one YAML document, rejecting a missing or non-mapping file."""
    target = Path(path)
    if not target.is_file():
        raise ConfigError(f"config file not found: {target}")
    with target.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"config file must hold a mapping: {target}")
    return loaded


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively overlay ``override`` on top of ``base`` without mutating either."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _coerce(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``a.b.c=value`` overrides, creating intermediate mappings as needed."""
    result = dict(config)
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"override must be key=value, got {item!r}")
        dotted, raw = item.split("=", 1)
        *parents, leaf = dotted.split(".")
        cursor = result
        for parent in parents:
            nxt = cursor.get(parent)
            if not isinstance(nxt, dict):
                nxt = {}
                cursor[parent] = nxt
            cursor = nxt
        cursor[leaf] = _coerce(raw)
    return result


def resolve_experiment(
    name: str,
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Load ``<config_dir>/experiment/<name>.yaml`` and its declared partials."""
    root = Path(config_dir)
    raw = load_yaml(root / "experiment" / f"{name}.yaml")
    partials = raw.pop("extends", {})
    composed: dict[str, Any] = {}
    if isinstance(partials, dict):
        for section in ("data", "model", "train"):
            reference = partials.get(section)
            if reference is None:
                continue
            composed = deep_merge(composed, {section: load_yaml(root / reference)})
    composed = deep_merge(composed, raw)
    if overrides:
        composed = apply_overrides(composed, overrides)
    composed.setdefault("experiment", {})
    composed["experiment"]["name"] = name
    return composed


def _require(mapping: dict[str, Any], key: str, where: str) -> Any:
    value = mapping.get(key, _MISSING)
    if value is _MISSING:
        raise ConfigError(f"missing required key {where}.{key}")
    return value


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, dict):
        raise ConfigError(f"section {name} must be a mapping")
    return value


@dataclass(frozen=True)
class CohortConfig:
    """Sites, regions, split sizes and the boundary rule that defines the endpoint."""

    sites: tuple[str, ...]
    regions: dict[str, str]
    development_sites: tuple[str, ...]
    external_sites: tuple[str, ...]
    train_size: int
    internal_test_size: int
    external_size: int
    prospective_size: int
    seed_count: int
    median_harvested_nodes: dict[str, int]
    adequate_node_yield: int
    scanner_vendors: tuple[str, ...]
    modalities: tuple[str, ...]

    @property
    def development_size(self) -> int:
        return self.train_size + self.internal_test_size

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> CohortConfig:
        return cls(
            sites=tuple(raw.get("sites", ["A", "B", "C", "D", "E"])),
            regions=dict(raw.get("regions", {"A": "I", "B": "I", "C": "II", "D": "III", "E": "III"})),
            development_sites=tuple(raw.get("development_sites", ["A", "B", "C"])),
            external_sites=tuple(raw.get("external_sites", ["D", "E"])),
            train_size=int(raw.get("train_size", 5868)),
            internal_test_size=int(raw.get("internal_test_size", 1960)),
            external_size=int(raw.get("external_size", 3456)),
            prospective_size=int(raw.get("prospective_size", 1850)),
            seed_count=int(raw.get("seed_count", 5)),
            median_harvested_nodes=dict(raw.get("median_harvested_nodes", {"A": 32, "B": 27, "C": 21, "D": 18, "E": 16})),
            adequate_node_yield=int(raw.get("adequate_node_yield", 16)),
            scanner_vendors=tuple(raw.get("scanner_vendors", ["Siemens", "GE", "Philips", "Toshiba"])),
            modalities=tuple(raw.get("modalities", ["ct", "radiomics", "clinical", "endoscopy", "pathology"])),
        )


@dataclass(frozen=True)
class ImagingConfig:
    """Image acquisition and preprocessing contract (Methods Sec. 4.3)."""

    spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0)
    hu_window: tuple[float, float] = (-150.0, 250.0)
    bias_field_sigma_mm: float = 40.0
    intensity_standardisation: str = "zscore"
    patch_grid: tuple[int, int, int] = (96, 96, 64)
    shell_inner_mm: float = 0.0
    shell_outer_mm: float = 5.0
    rotate_degrees: float = 5.0
    scale_fraction: float = 0.05
    intensity_jitter_hu: float = 20.0

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> ImagingConfig:
        window = raw.get("hu_window", [-150.0, 250.0])
        return cls(
            spacing_mm=tuple(float(v) for v in raw.get("spacing_mm", [1.0, 1.0, 1.0])),  # type: ignore[arg-type]
            hu_window=(float(window[0]), float(window[1])),
            bias_field_sigma_mm=float(raw.get("bias_field_sigma_mm", 40.0)),
            intensity_standardisation=str(raw.get("intensity_standardisation", "zscore")),
            patch_grid=tuple(int(v) for v in raw.get("patch_grid", [96, 96, 64])),  # type: ignore[arg-type]
            shell_inner_mm=float(raw.get("shell_inner_mm", 0.0)),
            shell_outer_mm=float(raw.get("shell_outer_mm", 5.0)),
            rotate_degrees=float(raw.get("rotate_degrees", 5.0)),
            scale_fraction=float(raw.get("scale_fraction", 0.05)),
            intensity_jitter_hu=float(raw.get("intensity_jitter_hu", 20.0)),
        )


@dataclass(frozen=True)
class RadiomicsConfig:
    """Peritumoral descriptor extraction, stability selection and redundancy collapse."""

    families: tuple[str, ...] = (
        "firstorder",
        "glcm",
        "glrlm",
        "glszm",
        "gldm",
        "ngtdm",
        "shape",
    )
    stability_repeats: int = 12
    stability_icc_threshold: float = 0.75
    redundancy_pearson_threshold: float = 0.90
    standardise_scope: str = "per_site"
    descriptor_cap: int = 64

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> RadiomicsConfig:
        return cls(
            families=tuple(raw.get("families", list(cls().families))),
            stability_repeats=int(raw.get("stability_repeats", 12)),
            stability_icc_threshold=float(raw.get("stability_icc_threshold", 0.75)),
            redundancy_pearson_threshold=float(raw.get("redundancy_pearson_threshold", 0.90)),
            standardise_scope=str(raw.get("standardise_scope", "per_site")),
            descriptor_cap=int(raw.get("descriptor_cap", 64)),
        )


@dataclass(frozen=True)
class EncoderConfig:
    """Frozen encoder and the low-rank adaptation applied to its attention projections."""

    name: str = "merlin"
    weights_path: str | None = None
    embed_dim: int = 384
    depth: int = 8
    num_heads: int = 6
    patch_size: int = 16
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "out_proj")
    freeze_backbone: bool = True

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> EncoderConfig:
        adapt = raw.get("adaptation", {})
        return cls(
            name=str(raw.get("name", "merlin")),
            weights_path=raw.get("weights_path"),
            embed_dim=int(raw.get("embed_dim", 384)),
            depth=int(raw.get("depth", 8)),
            num_heads=int(raw.get("num_heads", 6)),
            patch_size=int(raw.get("patch_size", 16)),
            lora_rank=int(adapt.get("rank", 16)),
            lora_alpha=int(adapt.get("alpha", 32)),
            lora_targets=tuple(adapt.get("targets", ["q_proj", "k_proj", "v_proj", "out_proj"])),
            freeze_backbone=bool(raw.get("freeze_backbone", True)),
        )


@dataclass(frozen=True)
class FusionConfig:
    """Cross-attention fusion over image tokens and the non-imaging streams."""

    embed_dim: int = 384
    num_heads: int = 6
    layers: int = 2
    dropout: float = 0.1
    modality_dropout: float = 0.15
    image_token_cap: int = 256
    presence_embedding: bool = True

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> FusionConfig:
        return cls(
            embed_dim=int(raw.get("embed_dim", 384)),
            num_heads=int(raw.get("num_heads", 6)),
            layers=int(raw.get("layers", 2)),
            dropout=float(raw.get("dropout", 0.1)),
            modality_dropout=float(raw.get("modality_dropout", 0.15)),
            image_token_cap=int(raw.get("image_token_cap", 256)),
            presence_embedding=bool(raw.get("presence_embedding", True)),
        )


@dataclass(frozen=True)
class OrdinalConfig:
    """The monotonic ordinal head shared by the three axes."""

    t_classes: int = 4
    n_classes: int = 4
    m_classes: int = 2
    cut_point_margin: float = 1e-3

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> OrdinalConfig:
        axes = raw.get("axes", {})
        return cls(
            t_classes=int(axes.get("t_classes", 4)),
            n_classes=int(axes.get("n_classes", 4)),
            m_classes=int(axes.get("m_classes", 2)),
            cut_point_margin=float(raw.get("cut_point_margin", 1e-3)),
        )


@dataclass(frozen=True)
class RiskConfig:
    """Site-conditional calibration and per-stratum threshold selection."""

    affine_scope: str = "site"
    threshold_grid: tuple[float, ...] = tuple(round(0.02 * step, 2) for step in range(2, 50))
    confidence: float = 0.05
    min_stratum_size: int = 24
    transfer_corrections: bool = True

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> RiskConfig:
        grid = raw.get("threshold_grid")
        if grid is None:
            grid = [round(0.02 * step, 2) for step in range(2, 50)]
        return cls(
            affine_scope=str(raw.get("affine_scope", "site")),
            threshold_grid=tuple(float(v) for v in grid),
            confidence=float(raw.get("confidence", 0.05)),
            min_stratum_size=int(raw.get("min_stratum_size", 24)),
            transfer_corrections=bool(raw.get("transfer_corrections", True)),
        )


@dataclass(frozen=True)
class LossConfig:
    """Weights of the three loss terms in Algorithm 1, step 9."""

    lambda_decision: float = 0.5
    gamma_calibration: float = 0.1
    ordinal_weighting: str = "cumulative"
    calibration_variant: str = "site_gap"

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> LossConfig:
        return cls(
            lambda_decision=float(raw.get("lambda_decision", 0.5)),
            gamma_calibration=float(raw.get("gamma_calibration", 0.1)),
            ordinal_weighting=str(raw.get("ordinal_weighting", "cumulative")),
            calibration_variant=str(raw.get("calibration_variant", "site_gap")),
        )


@dataclass(frozen=True)
class TrainConfig:
    """Optimisation schedule reported in Methods Sec. 4.7."""

    batch_size: int = 8
    grad_accum: int = 1
    epochs: int = 60
    early_stopping_patience: int = 8
    lr_fusion: float = 3e-4
    lr_adapter: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    scheduler: str = "cosine"
    grad_clip: float = 1.0
    precision: str = "bf16"
    world_size: int = 4
    accelerator_hours: float = 3100.0
    accelerator_memory_gb: float = 80.0
    num_workers: int = 4
    eval_every: int = 1
    checkpoint_every: int = 5
    ema_decay: float = 0.0
    use_ema: bool = False

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum * self.world_size

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> TrainConfig:
        optim = raw.get("optim", {})
        sched = raw.get("schedule", {})
        hardware = raw.get("hardware", {})
        return cls(
            batch_size=int(raw.get("batch_size", 8)),
            grad_accum=int(raw.get("grad_accum", 1)),
            epochs=int(raw.get("epochs", 60)),
            early_stopping_patience=int(raw.get("early_stopping_patience", 8)),
            lr_fusion=float(optim.get("lr_fusion", 3e-4)),
            lr_adapter=float(optim.get("lr_adapter", 1e-4)),
            weight_decay=float(optim.get("weight_decay", 1e-4)),
            warmup_steps=int(sched.get("warmup_steps", 500)),
            scheduler=str(sched.get("name", "cosine")),
            grad_clip=float(optim.get("grad_clip", 1.0)),
            precision=str(raw.get("precision", "bf16")),
            world_size=int(hardware.get("world_size", 4)),
            accelerator_hours=float(hardware.get("accelerator_hours", 3100.0)),
            accelerator_memory_gb=float(hardware.get("accelerator_memory_gb", 80.0)),
            num_workers=int(raw.get("num_workers", 4)),
            eval_every=int(raw.get("eval_every", 1)),
            checkpoint_every=int(raw.get("checkpoint_every", 5)),
            ema_decay=float(raw.get("ema_decay", 0.0)),
            use_ema=bool(raw.get("use_ema", False)),
        )


@dataclass(frozen=True)
class ExperimentConfig:
    """The composed config handed to a CLI entry point."""

    name: str
    cohort: CohortConfig
    imaging: ImagingConfig
    radiomics: RadiomicsConfig
    encoder: EncoderConfig
    fusion: FusionConfig
    ordinal: OrdinalConfig
    risk: RiskConfig
    loss: LossConfig
    train: TrainConfig
    seed: int = 0
    ablations: dict[str, bool] = field(default_factory=dict)
    streams: tuple[str, ...] = ("ct", "radiomics", "clinical", "endoscopy", "pathology")
    evaluation: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, config: dict[str, Any]) -> ExperimentConfig:
        data = _section(config, "data")
        model = _section(config, "model")
        train = _section(config, "train")
        experiment = _section(config, "experiment")
        cohort_raw = dict(data.get("cohort", {}))
        cohort = CohortConfig.from_mapping(cohort_raw)
        streams = tuple(experiment.get("streams", cohort.modalities))
        ablations = dict(experiment.get("ablations", {}))
        seed = int(experiment.get("seed", train.get("seed", 0)))
        configured_streams = set(cohort.modalities)
        unknown = sorted(set(ablations) - configured_streams - {"stage_consistency", "risk_control", "fusion", "radiomics", "adaptation", "ordinal"})
        if unknown:
            raise ConfigError(f"unknown ablation switch(es): {', '.join(unknown)}")
        return cls(
            name=str(experiment.get("name", "main")),
            cohort=cohort,
            imaging=ImagingConfig.from_mapping(dict(data.get("imaging", {}))),
            radiomics=RadiomicsConfig.from_mapping(dict(data.get("radiomics", {}))),
            encoder=EncoderConfig.from_mapping(dict(model.get("encoder", {}))),
            fusion=FusionConfig.from_mapping(dict(model.get("fusion", {}))),
            ordinal=OrdinalConfig.from_mapping(dict(model.get("ordinal", {}))),
            risk=RiskConfig.from_mapping(dict(model.get("risk", {}))),
            loss=LossConfig.from_mapping(dict(model.get("loss", {}))),
            train=TrainConfig.from_mapping(train),
            seed=seed,
            ablations=ablations,
            streams=streams,
            evaluation=dict(experiment.get("evaluation", {})),
        )

    def ablation_enabled(self, name: str) -> bool:
        """True when the named component is present in this arm."""
        return not bool(self.ablations.get(name, False))


def require(config: dict[str, Any], dotted: str) -> Any:
    """Fetch a dotted key from a nested mapping, raising when absent."""
    cursor: Any = config
    for part in dotted.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            raise ConfigError(f"missing required key {dotted}")
        cursor = cursor[part]
    return cursor
