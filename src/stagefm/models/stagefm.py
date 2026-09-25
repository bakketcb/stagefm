"""STAGEFM: the four modules assembled into one staging model.

The forward pass is the manuscript's pipeline in order. The encoder produces patch
tokens under low-rank adaptation, the fusion block combines them with the
non-imaging streams under modality dropout, the three monotonic ordinal heads give
one cumulative-link distribution per axis, those are composed into a joint
distribution over stage combinations and projected onto the achievable set, and the
projected distribution supplies the boundary logit the risk-control layer acts on.

Every structural component can be switched off, because each ablation row in the
manuscript's Table 4 removes exactly one of them. The switches change the support of
the label space and the decision boundary, never the representation, which is the
distinction the ablation is designed to expose.

Ref: Methods Sec. 4.5 (the four modules); Algorithm 1 (training and inference);
Sec. 4.6 (component justification).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from ..data.schema import M_CLASSES, N_CLASSES, T_CLASSES
from ..data.staging import AchievableSet
from ..data.streams import StreamPlan
from ..utils.config import ExperimentConfig
from .encoder import EncoderOutput, build_encoder, cap_tokens
from .fusion import FusionInputs, FusionOutput, build_fusion
from .lora import AdapterGroups, apply_lora, is_adapter_parameter, parameter_counts
from .ordinal import OrdinalOutput, StagingHeads
from .stage_consistency import FeasibilityProjection, joint_distribution


@dataclass(frozen=True)
class VariantFlags:
    """Which structural components this arm keeps."""

    stage_consistency: bool = True
    risk_control: bool = True
    fusion: bool = True
    ordinal: bool = True
    adaptation: bool = True
    use_image_tokens: bool = True
    fusion_strategy: str = "cross_attention"
    post_hoc_projection: bool = False

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> VariantFlags:
        flags = config.ablations
        return cls(
            stage_consistency=not bool(flags.get("stage_consistency", False)),
            risk_control=not bool(flags.get("risk_control", False)),
            fusion=not bool(flags.get("fusion", False)),
            ordinal=not bool(flags.get("ordinal", False)),
            adaptation=not bool(flags.get("adaptation", False)),
            use_image_tokens=not bool(flags.get("ct", False)),
            fusion_strategy=str(config.evaluation.get("fusion_strategy", "cross_attention")),
            post_hoc_projection=bool(config.evaluation.get("post_hoc_projection", False)),
        )


@dataclass(frozen=True)
class StageOutput:
    """Everything the losses, the metrics stage and the risk layer consume."""

    axis: dict[str, OrdinalOutput]
    joint: Tensor
    projected: Tensor
    fused: Tensor
    boundary_logit: Tensor
    feasibility_normaliser: Tensor


class STAGEFM(nn.Module):
    """The staging model."""

    def __init__(
        self,
        config: ExperimentConfig,
        plan: StreamPlan,
        achievable: AchievableSet,
        radiomics_dim: int,
        clinical_dim: int,
        endoscopy_vocab: int,
        pathology_vocab: int,
        flags: VariantFlags | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.plan = plan
        self.flags = flags or VariantFlags.from_config(config)
        self.encoder, self.encoder_metadata = build_encoder(config.encoder)
        if self.flags.adaptation:
            apply_lora(self.encoder, config.encoder.lora_rank, config.encoder.lora_alpha, config.encoder.lora_targets)
        if config.encoder.freeze_backbone:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)
            if self.flags.adaptation:
                for name, parameter in self.encoder.named_parameters():
                    if is_adapter_parameter(name):
                        parameter.requires_grad_(True)

        self.fusion: nn.Module | None
        if self.flags.fusion:
            self.fusion = build_fusion(
                self.flags.fusion_strategy,
                config.fusion,
                radiomics_dim=radiomics_dim,
                clinical_dim=clinical_dim,
                endoscopy_vocab=endoscopy_vocab,
                pathology_vocab=pathology_vocab,
                active_streams=plan.non_imaging,
                use_image_tokens=self.flags.use_image_tokens,
            )
        else:
            self.fusion = None
        embed_dim = config.fusion.embed_dim
        self.probe_norm = nn.LayerNorm(embed_dim)
        self.heads = StagingHeads(embed_dim, config.ordinal, ordinal=self.flags.ordinal)
        self.projection = FeasibilityProjection(achievable)

    def encode(self, volume: Tensor) -> EncoderOutput:
        output = self.encoder(volume)
        return EncoderOutput(tokens=cap_tokens(output.tokens, self.config.fusion.image_token_cap), grid=output.grid)

    def fuse(self, tokens: Tensor, batch: dict[str, Any]) -> FusionOutput:
        if self.fusion is None or not self.flags.fusion:
            pooled = self.probe_norm(tokens.mean(dim=1))
            return FusionOutput(fused=pooled, context=pooled.unsqueeze(1), stream_tokens=pooled.unsqueeze(1))
        inputs = FusionInputs(
            image_tokens=tokens,
            radiomics=batch["radiomics"],
            clinical=batch["clinical"],
            endoscopy=batch["endoscopy"],
            pathology=batch["pathology"],
            presence=batch["presence"],
        )
        return self.fusion(inputs)  # type: ignore[no-any-return]

    def forward(self, batch: dict[str, Any]) -> StageOutput:
        encoded = self.encode(batch["volume"])
        fused = self.fuse(encoded.tokens, batch)
        axis = self.heads(fused.fused)
        joint = joint_distribution(axis["T"].probabilities, axis["N"].probabilities, axis["M"].probabilities)
        if self.flags.stage_consistency:
            projection = self.projection(joint)
            projected = projection.probabilities
            normaliser = projection.normaliser
        elif self.flags.post_hoc_projection:
            # The unconstrained control: the loss sees the raw joint, only the
            # reported distribution is made feasible, after the fact.
            projection = self.projection(joint)
            projected = projection.probabilities
            normaliser = projection.normaliser
        else:
            projected = joint
            normaliser = joint.sum(dim=-1)
        boundary_logit = systemic_logit(projected)
        return StageOutput(
            axis=axis,
            joint=joint,
            projected=projected,
            fused=fused.fused,
            boundary_logit=boundary_logit,
            feasibility_normaliser=normaliser,
        )

    def components(self) -> dict[str, int]:
        """Parameter counts of each module, for the run metadata."""
        return {
            "encoder": sum(parameter.numel() for parameter in self.encoder.parameters()),
            "fusion": sum(parameter.numel() for parameter in self.fusion.parameters()) if self.fusion is not None else 0,
            "heads": sum(parameter.numel() for parameter in self.heads.parameters()),
        }

    def adapter_report(self) -> dict[str, float]:
        return parameter_counts(self.encoder)

    def trainable_groups(self) -> AdapterGroups:
        return AdapterGroups(self)

    def monotonicity_violations(self) -> int:
        return self.heads.monotonicity_violations()


def systemic_logit(projected: Tensor) -> Tensor:
    """Log-odds that the management category is systemic, from the projected joint.

    The category is read off the projected distribution rather than from a separate
    head, so the decision the risk layer adjusts and the stage the model reports can
    never disagree.
    """
    mapping = _category_mapping(projected.shape[-1], projected.device)
    systemic = (projected * (mapping == 2).to(projected.dtype)).sum(dim=-1)
    earlier = 1.0 - systemic
    return torch.log(systemic.clamp(min=1e-9)) - torch.log(earlier.clamp(min=1e-9))


_CATEGORY_CACHE: dict[tuple[int, str], Tensor] = {}


def _category_mapping(count: int, device: torch.device) -> Tensor:
    """Category index of each flat stage column, cached per device."""
    key = (count, str(device))
    cached = _CATEGORY_CACHE.get(key)
    if cached is not None:
        return cached
    from ..data.schema import ALL_TRIPLES
    from ..data.staging import treatment_category

    order = {"surgery_first": 0, "perioperative": 1, "systemic": 2}
    mapping = torch.tensor([order[treatment_category(stage).value] for stage in ALL_TRIPLES[:count]], dtype=torch.long, device=device)
    _CATEGORY_CACHE[key] = mapping
    return mapping


def predicted_columns(projected: np.ndarray) -> np.ndarray:
    """Argmax decode of a projected joint distribution."""
    return np.asarray(projected.argmax(axis=-1), dtype=np.int64)


def cardinalities() -> tuple[int, int, int]:
    return (T_CLASSES, N_CLASSES, M_CLASSES)
