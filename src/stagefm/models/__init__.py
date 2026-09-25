"""Model modules: encoder, low-rank adaptation, fusion, heads, constraints, arms."""

from __future__ import annotations

from .baselines import BASELINES, QUOTED_ARMS, BaselineSpec, StreamDimensions, StreamOnlyModel, arm_registry, arm_table_rows, build_arm
from .encoder import CompactCTEncoder, EncoderOutput, EncoderUnavailable, build_encoder
from .fusion import CrossAttentionFusion, EarlyConcatFusion, FusionInputs, FusionOutput, LateAverageFusion, build_fusion
from .lora import AdapterGroups, LoRALinear, adapter_parameter_count, apply_lora, freeze_except, is_adapter_parameter, parameter_counts
from .ordinal import MonotonicOrdinalHead, OrdinalOutput, SoftmaxHead, StagingHeads
from .risk_control import (
    AffineCorrection,
    SiteCalibrator,
    ThresholdSelection,
    apply_thresholds,
    category_masses,
    decide_stage_category,
    reference_categories,
    select_thresholds,
)
from .stage_consistency import FeasibilityProjection, ProjectionOutput, joint_distribution
from .stagefm import STAGEFM, StageOutput, VariantFlags, predicted_columns, systemic_logit

__all__ = [
    "BASELINES",
    "QUOTED_ARMS",
    "STAGEFM",
    "AdapterGroups",
    "AffineCorrection",
    "BaselineSpec",
    "CompactCTEncoder",
    "CrossAttentionFusion",
    "EarlyConcatFusion",
    "EncoderOutput",
    "EncoderUnavailable",
    "FeasibilityProjection",
    "FusionInputs",
    "FusionOutput",
    "LoRALinear",
    "LateAverageFusion",
    "MonotonicOrdinalHead",
    "OrdinalOutput",
    "ProjectionOutput",
    "QUOTED_ARMS",
    "SiteCalibrator",
    "SoftmaxHead",
    "StageOutput",
    "StagingHeads",
    "StreamDimensions",
    "StreamOnlyModel",
    "ThresholdSelection",
    "VariantFlags",
    "adapter_parameter_count",
    "apply_lora",
    "freeze_except",
    "is_adapter_parameter",
    "apply_thresholds",
    "arm_registry",
    "arm_table_rows",
    "build_arm",
    "build_encoder",
    "build_fusion",
    "category_masses",
    "decide_stage_category",
    "joint_distribution",
    "parameter_counts",
    "predicted_columns",
    "reference_categories",
    "select_thresholds",
    "systemic_logit",
]
