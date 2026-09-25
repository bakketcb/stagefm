"""Comparison arms.

The manuscript evaluates eleven arms on the same external cohort with the same
tuning budget. Eight of them are implemented here and trained by the same driver as
the model; the two clinical reference points are quoted from their sources and are
never recomputed, and the published multicentre range is carried as a quoted value
so no arm-level number in a results table can be mistaken for a measurement this
code produced.

The implemented arms are: a task-specific CT network, a radiomics plus clinical
logistic model, a linear probe on the frozen encoder, a fully fine-tuned encoder
with independent T/N/M heads, a multimodal 3D network without pre-training, a
report-text-only model, a structured-clinical-only model, and the unconstrained
model whose feasibility is projected after the fact.

Ref: Methods Sec. 4.8 (baselines); Table 3; Fig. 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import torch
from torch import Tensor, nn

from ..data.staging import AchievableSet
from ..data.streams import StreamPlan
from ..utils.config import ExperimentConfig
from .fusion import FusionInputs, build_fusion
from .ordinal import StagingHeads
from .stage_consistency import FeasibilityProjection, joint_distribution
from .stagefm import STAGEFM, StageOutput, VariantFlags, systemic_logit


@dataclass(frozen=True)
class StreamDimensions:
    """Input widths and vocabulary sizes the fusion block needs."""

    radiomics: int
    clinical: int
    endoscopy_vocab: int
    pathology_vocab: int


@dataclass(frozen=True)
class BaselineSpec:
    """One comparison arm."""

    key: str
    label: str
    description: str
    kind: str
    flags: VariantFlags | None = None
    streams: tuple[str, ...] = ("radiomics", "clinical", "endoscopy", "pathology")
    tunes_all_encoder_weights: bool = False
    quoted_metrics: dict[str, float] = field(default_factory=dict)
    source: str = ""


@dataclass(frozen=True)
class QuotedArm:
    """An arm whose values come from a source outside this code."""

    key: str
    label: str
    metrics: dict[str, float]
    source: str


QUOTED_ARMS: dict[str, QuotedArm] = {
    "eus": QuotedArm(
        key="eus",
        label="Endoscopic ultrasound reading",
        metrics={"t_accuracy": 0.684, "nodal_sensitivity": 0.621, "discordance": 11.82},
        source="clinical standard reported in the manuscript, Ref. Sec. 2.2 and Fig. 1",
    ),
    "reader_panel": QuotedArm(
        key="reader_panel",
        label="Human reader panel, unassisted",
        metrics={"t_accuracy": 0.617, "accuracy_low": 0.579, "accuracy_high": 0.655, "weighted_kappa": 0.44, "discordance": 16.45},
        source="reader study reported in the manuscript, Sec. 2.6 and Sec. 4.9",
    ),
    "reader_panel_assisted": QuotedArm(
        key="reader_panel_assisted",
        label="Human reader panel, model-assisted",
        metrics={"t_accuracy": 0.783, "discordance": 7.28},
        source="reader study reported in the manuscript, Sec. 3",
    ),
    "published_multicentre_range": QuotedArm(
        key="published_multicentre_range",
        label="Published multicentre T-axis range",
        metrics={"t_auroc_low": 0.91, "t_auroc_high": 0.95, "accuracy_low": 0.87, "accuracy_high": 0.94},
        source="multicentre study quoted in the manuscript, Ref. Sec. 1",
    ),
}


BASELINES: dict[str, BaselineSpec] = {
    "task_specific_ct": BaselineSpec(
        key="task_specific_ct",
        label="Task-specific CT backbone",
        description="A backbone trained from scratch for this task instead of adapting a frozen encoder.",
        kind="stagefm_variant",
        flags=VariantFlags(
            stage_consistency=True,
            risk_control=True,
            fusion=True,
            ordinal=True,
            adaptation=False,
            fusion_strategy="cross_attention",
        ),
        tunes_all_encoder_weights=True,
    ),
    "radiomics_clinical_logistic": BaselineSpec(
        key="radiomics_clinical_logistic",
        label="Radiomics plus clinical logistic model",
        description="Logistic model on the peritumoral descriptors and the structured clinical covariates.",
        kind="streams_only",
        streams=("radiomics", "clinical"),
    ),
    "frozen_linear_probe": BaselineSpec(
        key="frozen_linear_probe",
        label="Frozen encoder with a linear probe",
        description="The released representation with no adaptation and no fusion; the probe reads pooled image tokens only.",
        kind="stagefm_variant",
        flags=VariantFlags(stage_consistency=True, risk_control=True, fusion=False, ordinal=True, adaptation=False),
    ),
    "finetuned_independent_heads": BaselineSpec(
        key="finetuned_independent_heads",
        label="Fine-tuned encoder with independent heads",
        description="Full fine-tuning with independent T, N and M heads and no feasibility enforcement.",
        kind="stagefm_variant",
        flags=VariantFlags(stage_consistency=False, risk_control=True, fusion=True, ordinal=False, adaptation=False),
        tunes_all_encoder_weights=True,
    ),
    "multimodal_3d_no_pretraining": BaselineSpec(
        key="multimodal_3d_no_pretraining",
        label="Multimodal 3D network without pre-training",
        description="The same fusion and heads over an encoder trained from scratch, with no pre-trained representation.",
        kind="stagefm_variant",
        flags=VariantFlags(stage_consistency=True, risk_control=True, fusion=True, ordinal=True, adaptation=False),
        tunes_all_encoder_weights=True,
    ),
    "report_text_only": BaselineSpec(
        key="report_text_only",
        label="Report text only",
        description="The text streams alone, without imaging or structured covariates.",
        kind="streams_only",
        streams=("endoscopy", "pathology"),
    ),
    "structured_clinical_only": BaselineSpec(
        key="structured_clinical_only",
        label="Structured clinical data only",
        description="The structured clinical covariates alone.",
        kind="streams_only",
        streams=("clinical",),
    ),
    "unconstrained_posthoc": BaselineSpec(
        key="unconstrained_posthoc",
        label="Unconstrained with post-hoc feasibility projection",
        description="Independent heads, no feasibility constraint during training, and the same projection applied to the output distribution after the fact.",
        kind="stagefm_variant",
        flags=VariantFlags(
            stage_consistency=False,
            risk_control=False,
            fusion=True,
            ordinal=False,
            adaptation=False,
            post_hoc_projection=True,
        ),
        tunes_all_encoder_weights=True,
    ),
}


class StreamOnlyModel(nn.Module):
    """A model that reads only a fixed subset of the non-imaging streams.

    Streams are kept on the same token contract as the full model so the trainer and
    the metrics stage do not branch on the arm.
    """

    def __init__(self, config: ExperimentConfig, streams: tuple[str, ...], dimensions: StreamDimensions, achievable: AchievableSet) -> None:
        super().__init__()
        self.config = config
        self.streams = streams
        self.achievable = achievable
        from ..data.schema import Stream

        active = tuple(Stream(name) for name in streams)
        self.plan_dummy = StreamPlan(active=active)
        self.fusion = build_fusion(
            "cross_attention",
            config.fusion,
            radiomics_dim=dimensions.radiomics,
            clinical_dim=dimensions.clinical,
            endoscopy_vocab=dimensions.endoscopy_vocab,
            pathology_vocab=dimensions.pathology_vocab,
            active_streams=active,
            use_image_tokens=False,
        )
        self.heads = StagingHeads(config.fusion.embed_dim, config.ordinal, ordinal=True)
        self.projection = FeasibilityProjection(achievable)
        self.flags = VariantFlags()
        self.encoder_metadata: dict[str, object] = {"encoder": "none", "weights_path": None, "embed_dim": config.fusion.embed_dim}

    def forward(self, batch: dict[str, Tensor]) -> StageOutput:
        image_tokens = torch.zeros(
            batch["radiomics"].shape[0],
            0,
            self.config.fusion.embed_dim,
            device=batch["radiomics"].device,
            dtype=batch["radiomics"].dtype,
        )
        inputs = FusionInputs(
            image_tokens=image_tokens,
            radiomics=batch["radiomics"],
            clinical=batch["clinical"],
            endoscopy=batch["endoscopy"],
            pathology=batch["pathology"],
            presence=batch["presence"],
        )
        fused = self.fusion(inputs)
        axis = self.heads(fused.fused)
        joint = joint_distribution(axis["T"].probabilities, axis["N"].probabilities, axis["M"].probabilities)
        projection = self.projection(joint)
        return StageOutput(
            axis=axis,
            joint=joint,
            projected=projection.probabilities,
            fused=fused.fused,
            boundary_logit=systemic_logit(projection.probabilities),
            feasibility_normaliser=projection.normaliser,
        )

    def components(self) -> dict[str, int]:
        return {"encoder": 0, "fusion": sum(p.numel() for p in self.fusion.parameters()), "heads": sum(p.numel() for p in self.heads.parameters())}

    def trainable_groups(self) -> object:
        from .lora import AdapterGroups

        return AdapterGroups(self)

    def adapter_report(self) -> dict[str, float]:
        return {"encoder_total": 0.0, "adapter_total": 0.0, "adapter_fraction": 0.0}

    def monotonicity_violations(self) -> int:
        return self.heads.monotonicity_violations()


def build_arm(
    key: str,
    config: ExperimentConfig,
    plan: StreamPlan,
    achievable: AchievableSet,
    dimensions: StreamDimensions,
) -> nn.Module:
    """Construct the arm named by ``key``."""
    if key == "stagefm" or key == "main":
        return STAGEFM(
            config,
            plan,
            achievable,
            radiomics_dim=dimensions.radiomics,
            clinical_dim=dimensions.clinical,
            endoscopy_vocab=dimensions.endoscopy_vocab,
            pathology_vocab=dimensions.pathology_vocab,
        )
    if key not in BASELINES:
        raise KeyError(f"unknown arm: {key}")
    spec = BASELINES[key]
    if spec.kind == "streams_only":
        return StreamOnlyModel(config, spec.streams, dimensions, achievable)
    if spec.kind == "stagefm_variant":
        return STAGEFM(
            config,
            plan,
            achievable,
            radiomics_dim=dimensions.radiomics,
            clinical_dim=dimensions.clinical,
            endoscopy_vocab=dimensions.endoscopy_vocab,
            pathology_vocab=dimensions.pathology_vocab,
            flags=spec.flags,
        )
    raise ValueError(f"arm {key} is quoted and cannot be constructed")


def arm_registry() -> dict[str, BaselineSpec]:
    """All trainable arms plus the quoted reference points, keyed by arm id."""
    registry = {"stagefm": BaselineSpec(key="stagefm", label="STAGEFM", description="The full model.", kind="stagefm_variant")}
    registry.update(BASELINES)
    return registry


def arm_table_rows() -> list[dict[str, object]]:
    """Rows for the results table: implemented arms first, quoted arms after."""
    rows: list[dict[str, object]] = []
    for key, spec in arm_registry().items():
        rows.append({"key": key, "label": spec.label, "source": "computed", "kind": spec.kind})
    for key, arm in QUOTED_ARMS.items():
        rows.append({"key": key, "label": arm.label, "source": "quoted", "kind": "quoted", "values": arm.metrics, "reference": arm.source})
    return rows


def with_projection(projected: Tensor, projection: FeasibilityProjection) -> Tensor:
    """Apply an existing feasibility projection to an already-computed distribution."""
    out: Tensor = projection(projected).probabilities
    return out


def variant_of(key: str) -> VariantFlags:
    """Flags of an arm, defaulting to the full model."""
    spec = BASELINES.get(key)
    if spec is None or spec.flags is None:
        return VariantFlags()
    return replace(spec.flags)
