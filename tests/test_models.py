"""Model tests: adaptation, encoder, fusion, ordinal heads, projection, risk control, arms."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from stagefm.data.schema import ALL_TRIPLES, Stream, TreatmentCategory, label_matrix
from stagefm.data.staging import AchievableSet, treatment_category
from stagefm.data.streams import StreamPlan
from stagefm.models.baselines import BASELINES, QUOTED_ARMS, StreamOnlyModel, arm_registry, arm_table_rows, build_arm
from stagefm.models.encoder import CompactCTEncoder, EncoderUnavailable, TransformerBlock, build_encoder, cap_tokens
from stagefm.models.fusion import CrossAttentionFusion, EarlyConcatFusion, FusionInputs, LateAverageFusion, build_fusion
from stagefm.models.lora import AdapterGroups, LoRALinear, adapter_parameter_count, apply_lora, freeze_except, parameter_counts
from stagefm.models.ordinal import MonotonicOrdinalHead, SoftmaxHead, StagingHeads
from stagefm.models.risk_control import (
    AffineCorrection,
    SiteCalibrator,
    apply_thresholds,
    category_masses,
    decide_stage_category,
    reference_categories,
    select_thresholds,
)
from stagefm.models.stage_consistency import FeasibilityProjection, joint_distribution
from stagefm.models.stagefm import VariantFlags, predicted_columns, systemic_logit
from stagefm.utils.config import EncoderConfig, ExperimentConfig, FusionConfig, OrdinalConfig, RiskConfig, resolve_experiment


def _inputs(batch: int, embed_dim: int, streams: tuple[Stream, ...] = (Stream.RADIOMICS, Stream.CLINICAL, Stream.ENDOSCOPY, Stream.PATHOLOGY)) -> FusionInputs:
    return FusionInputs(
        image_tokens=torch.randn(batch, 12, embed_dim),
        radiomics=torch.randn(batch, 8),
        clinical=torch.randn(batch, 5),
        endoscopy=torch.randint(0, 4, (batch, 6)),
        pathology=torch.randint(0, 4, (batch, 6)),
        presence=torch.ones(batch, len(streams)),
    )


def test_lora_wraps_targets_and_leaves_other_linears_alone() -> None:
    encoder = CompactCTEncoder(EncoderConfig(embed_dim=32, depth=2, num_heads=4, patch_size=4))
    replaced = apply_lora(encoder, rank=2, alpha=4, targets=("q_proj", "k_proj"))
    assert replaced == 4
    wrapped = [name for name, module in encoder.named_modules() if isinstance(module, LoRALinear)]
    assert len(wrapped) == 4
    assert all(name.endswith(("q_proj", "k_proj")) for name in wrapped)


def test_lora_rejects_targets_that_do_not_exist() -> None:
    encoder = CompactCTEncoder(EncoderConfig(embed_dim=32, depth=1, num_heads=4, patch_size=4))
    with pytest.raises(ValueError):
        apply_lora(encoder, rank=2, alpha=4, targets=("nope_proj",))


def test_lora_is_identity_at_initialisation_and_scales_by_alpha_over_rank() -> None:
    torch.manual_seed(0)
    layer = torch.nn.Linear(6, 4)
    wrapped = LoRALinear(layer, rank=2, alpha=4)
    inputs = torch.randn(3, 6)
    assert torch.allclose(wrapped(inputs), layer(inputs), atol=1e-6)
    assert wrapped.scaling == pytest.approx(2.0)
    with torch.no_grad():
        wrapped.lora_b.normal_()
    assert not torch.allclose(wrapped(inputs), layer(inputs))


def test_lora_freezes_the_base_layer() -> None:
    wrapped = LoRALinear(torch.nn.Linear(5, 3), rank=2, alpha=2)
    assert not any(parameter.requires_grad for parameter in wrapped.base.parameters())
    assert wrapped.lora_a.requires_grad and wrapped.lora_b.requires_grad
    assert adapter_parameter_count(wrapped) == 2 * 5 + 3 * 2


def test_freeze_except_and_adapter_groups() -> None:
    encoder = CompactCTEncoder(EncoderConfig(embed_dim=24, depth=1, num_heads=3, patch_size=4))
    apply_lora(encoder, rank=2, alpha=4, targets=("q_proj", "v_proj"))
    freeze_except(encoder, ("lora_",))
    groups = AdapterGroups(encoder)
    assert groups.adapter and not groups.other
    listed = groups.as_groups(lr_adapter=1e-4, lr_other=3e-4, weight_decay=1e-4)
    assert len(listed) == 1 and listed[0]["lr"] == pytest.approx(1e-4)


def test_parameter_counts_report_an_adapter_share() -> None:
    encoder = CompactCTEncoder(EncoderConfig(embed_dim=48, depth=2, num_heads=4, patch_size=4))
    apply_lora(encoder, rank=4, alpha=8, targets=("q_proj", "k_proj", "v_proj", "out_proj"))
    counts = parameter_counts(encoder)
    assert 0.0 < counts["adapter_fraction"] < 0.2
    assert counts["adapter_total"] == 8 * (2 * 4 * 48)


def test_encoder_output_shape_and_grid() -> None:
    encoder = CompactCTEncoder(EncoderConfig(embed_dim=32, depth=2, num_heads=4, patch_size=4))
    output = encoder(torch.randn(2, 1, 24, 16, 12))
    assert output.grid == (6, 4, 3)
    assert output.tokens.shape == (2, 72, 32)


def test_encoder_rejects_a_wrong_input_rank() -> None:
    encoder = CompactCTEncoder(EncoderConfig(embed_dim=16, depth=1, num_heads=2, patch_size=4))
    with pytest.raises(ValueError):
        encoder(torch.randn(2, 24, 16, 12))


def test_build_encoder_falls_back_and_records_the_choice() -> None:
    encoder, metadata = build_encoder(EncoderConfig(embed_dim=16, depth=1, num_heads=2))
    assert isinstance(encoder, CompactCTEncoder)
    assert metadata["encoder"] == "compact-3d-vit"
    assert metadata["weights_path"] is None


def test_build_encoder_reports_a_missing_checkpoint() -> None:
    from stagefm.models.encoder import MerlinEncoder

    with pytest.raises(EncoderUnavailable):
        MerlinEncoder(EncoderConfig(weights_path="/nonexistent/merlin.pt"))


def test_cap_tokens_truncates_in_grid_order() -> None:
    tokens = torch.arange(2 * 10 * 3, dtype=torch.float32).reshape(2, 10, 3)
    capped = cap_tokens(tokens, 4)
    assert capped.shape == (2, 4, 3)
    assert torch.equal(capped, tokens[:, :4, :])


def test_transformer_block_preserves_shape() -> None:
    block = TransformerBlock(embed_dim=16, num_heads=4)
    tokens = torch.randn(3, 7, 16)
    assert block(tokens).shape == tokens.shape


def test_cross_attention_fusion_shape_and_presence_effect() -> None:
    fusion = CrossAttentionFusion(
        config=FusionConfig(embed_dim=32, num_heads=4, layers=1),
        radiomics_dim=8,
        clinical_dim=5,
        endoscopy_vocab=4,
        pathology_vocab=4,
        active_streams=(Stream.RADIOMICS, Stream.CLINICAL, Stream.ENDOSCOPY, Stream.PATHOLOGY),
    )
    inputs = _inputs(3, 32)
    output = fusion(inputs)
    assert output.fused.shape == (3, 32)
    assert output.stream_tokens.shape == (3, 4, 32)
    absent = FusionInputs(
        image_tokens=inputs.image_tokens,
        radiomics=inputs.radiomics,
        clinical=inputs.clinical,
        endoscopy=inputs.endoscopy,
        pathology=inputs.pathology,
        presence=torch.zeros_like(inputs.presence),
    )
    assert not torch.allclose(fusion(inputs).fused, fusion(absent).fused)


def test_fusion_respects_the_active_stream_set() -> None:
    fusion = CrossAttentionFusion(
        config=FusionConfig(embed_dim=24, num_heads=4, layers=1),
        radiomics_dim=8,
        clinical_dim=5,
        endoscopy_vocab=4,
        pathology_vocab=4,
        active_streams=(Stream.CLINICAL,),
    )
    output = fusion(_inputs(2, 24))
    assert output.stream_tokens.shape == (2, 1, 24)


def test_fusion_strategy_registry() -> None:
    common = {
        "config": FusionConfig(embed_dim=24, num_heads=4, layers=1),
        "radiomics_dim": 8,
        "clinical_dim": 5,
        "endoscopy_vocab": 4,
        "pathology_vocab": 4,
        "active_streams": (Stream.RADIOMICS, Stream.CLINICAL, Stream.ENDOSCOPY, Stream.PATHOLOGY),
    }
    assert isinstance(build_fusion("cross_attention", **common), CrossAttentionFusion)  # type: ignore[arg-type]
    assert isinstance(build_fusion("early_concat", **common), EarlyConcatFusion)  # type: ignore[arg-type]
    assert isinstance(build_fusion("late_average", **common), LateAverageFusion)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        build_fusion("no_such_strategy", **common)  # type: ignore[arg-type]


def test_early_concat_and_late_average_produce_a_fused_vector() -> None:
    common = {
        "config": FusionConfig(embed_dim=24, num_heads=4, layers=1),
        "radiomics_dim": 8,
        "clinical_dim": 5,
        "endoscopy_vocab": 4,
        "pathology_vocab": 4,
        "active_streams": (Stream.RADIOMICS, Stream.CLINICAL, Stream.ENDOSCOPY, Stream.PATHOLOGY),
    }
    inputs = _inputs(2, 24)
    early = build_fusion("early_concat", **common)  # type: ignore[arg-type]
    late = build_fusion("late_average", **common)  # type: ignore[arg-type]
    assert early(inputs).fused.shape == (2, 24)
    assert late(inputs).stream_tokens.shape[1] == 4


def test_ordinal_head_probabilities_are_ordered_and_normalised() -> None:
    head = MonotonicOrdinalHead(input_dim=4, num_classes=4)
    features = torch.randn(50, 4)
    output = head(features)
    assert torch.allclose(output.probabilities.sum(dim=-1), torch.ones(50))
    cumulative = torch.sigmoid(output.cumulative_logits)
    assert bool((cumulative[:, 1:] <= cumulative[:, :-1] + 1e-6).all())
    assert head.cut_points.diff().min() > 0


def test_cut_points_stay_ordered_under_extreme_parameters() -> None:
    head = MonotonicOrdinalHead(input_dim=3, num_classes=5)
    for scale in (-50.0, 0.0, 50.0):
        with torch.no_grad():
            head.raw_gaps.fill_(scale)
        assert head.cut_points.diff().min() > 0
        assert bool(torch.all(head.cut_points[1:] > head.cut_points[:-1]))


def test_ordinal_head_rejects_a_single_category() -> None:
    with pytest.raises(ValueError):
        MonotonicOrdinalHead(input_dim=3, num_classes=1)


def test_softmax_head_is_available_as_the_unordered_alternative() -> None:
    head = SoftmaxHead(input_dim=4, num_classes=3)
    output = head(torch.randn(7, 4))
    assert output.probabilities.shape == (7, 3)
    assert output.cut_points.numel() == 0


def test_staging_heads_report_violations_only_for_ordered_heads() -> None:
    ordinal = StagingHeads(input_dim=8, config=OrdinalConfig(), ordinal=True)
    unordered = StagingHeads(input_dim=8, config=OrdinalConfig(), ordinal=False)
    features = torch.randn(5, 8)
    assert ordinal(features)["T"].probabilities.shape == (5, 4)
    assert ordinal.monotonicity_violations() == 0
    assert unordered.monotonicity_violations() == 0
    assert unordered.cut_points()["T"].numel() == 0


def test_joint_distribution_is_the_outer_product() -> None:
    t = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    n = torch.tensor([[0.4, 0.3, 0.2, 0.1]])
    m = torch.tensor([[0.7, 0.3]])
    joint = joint_distribution(t, n, m)
    assert joint.shape == (1, 32)
    assert float(joint.sum()) == pytest.approx(1.0)
    assert float(joint[0, 0]) == pytest.approx(0.1 * 0.4 * 0.7)


def test_joint_distribution_rejects_wrong_cardinalities() -> None:
    with pytest.raises(ValueError):
        joint_distribution(torch.rand(1, 3), torch.rand(1, 4), torch.rand(1, 2))


def test_projection_support_and_normalisation_against_enumeration() -> None:
    achievable = AchievableSet()
    projection = FeasibilityProjection(achievable)
    rng = np.random.default_rng(0)
    raw = rng.random((4, 32))
    raw = raw / raw.sum(axis=-1, keepdims=True)
    with torch.no_grad():
        output = projection(torch.tensor(raw, dtype=torch.float64))
    projected = output.probabilities.numpy()
    mask = np.asarray(achievable.mask, dtype=bool)
    expected = np.where(mask, raw, 0.0)
    expected = expected / expected.sum(axis=-1, keepdims=True)
    assert np.allclose(projected, expected)
    assert np.allclose(projected.sum(axis=-1), 1.0)
    assert np.allclose(projected[:, ~mask], 0.0)


def test_projection_degenerate_case_returns_uniform_on_the_achievable_set() -> None:
    achievable = AchievableSet()
    projection = FeasibilityProjection(achievable)
    mask = np.asarray(achievable.mask, dtype=bool)
    raw = np.where(mask, 0.0, 1.0)
    raw = raw / raw.sum()
    with torch.no_grad():
        output = projection(torch.tensor(raw, dtype=torch.float64).view(1, -1))
    assert bool(output.degenerate[0])
    assert np.allclose(output.probabilities.numpy()[0], mask / mask.sum())


def test_projection_rejects_a_wrong_width() -> None:
    projection = FeasibilityProjection(AchievableSet())
    with pytest.raises(ValueError):
        projection(torch.rand(1, 31))


def test_feasibility_predicate_matches_the_achievable_set() -> None:
    projection = FeasibilityProjection(AchievableSet())
    columns = torch.arange(32)
    assert int(projection.is_feasible(columns).sum()) == 22


def test_predicted_columns_decodes_the_argmax() -> None:
    probabilities = np.zeros((3, 32))
    probabilities[0, 5] = 1.0
    probabilities[1, 31] = 1.0
    probabilities[2, 0] = 1.0
    assert list(predicted_columns(probabilities)) == [5, 31, 0]


def test_systemic_logit_follows_the_category_mass() -> None:
    probabilities = np.zeros((2, 32))
    mapping = np.array([list(TreatmentCategory).index(treatment_category(stage)) for stage in ALL_TRIPLES])
    probabilities[0, np.flatnonzero(mapping == 2)[0]] = 1.0
    probabilities[1, np.flatnonzero(mapping == 0)[0]] = 1.0
    logits = systemic_logit(torch.tensor(probabilities))
    assert float(logits[0]) > 0
    assert float(logits[1]) < 0


def test_category_masses_decompose_the_joint() -> None:
    probabilities = np.full((1, 32), 1.0 / 32.0)
    masses = category_masses(probabilities)
    assert masses.shape == (1, 3)
    assert float(masses.sum()) == pytest.approx(1.0)


def test_decide_stage_category_respects_the_threshold() -> None:
    masses = np.array([[0.1, 0.1, 0.8], [0.6, 0.3, 0.1]])
    decided = decide_stage_category(masses, np.array([0.5, 0.5]))
    assert list(decided) == [2, 0]


def test_threshold_selection_minimises_the_empirical_error_and_bounds_it() -> None:
    masses = np.zeros((80, 3))
    masses[:, 0] = 0.6
    masses[:, 1] = 0.2
    masses[:, 2] = 0.2
    masses[::2, 2] = 0.5
    reference = np.where(np.arange(80) % 2 == 0, 2, 1)
    grid = (0.1, 0.3, 0.5, 0.7)
    selections = select_thresholds(masses, reference, ["A"] * 80, RiskConfig(threshold_grid=grid, min_stratum_size=4))
    chosen = selections["A"]
    errors = [float((decide_stage_category(masses, np.full(80, value)) != reference).mean()) for value in grid]
    assert chosen.empirical_error == pytest.approx(min(errors))
    assert chosen.bound >= chosen.empirical_error
    assert chosen.threshold in grid


def test_threshold_selection_flags_a_small_stratum() -> None:
    masses = np.zeros((5, 3))
    masses[:, 0] = 1.0
    selections = select_thresholds(masses, np.zeros(5, dtype=int), ["Z"] * 5, RiskConfig(min_stratum_size=24))
    assert not np.isfinite(selections["Z"].empirical_error)
    assert selections["Z"].threshold == 0.5


def test_apply_thresholds_uses_the_frozen_values() -> None:
    from stagefm.models.risk_control import ThresholdSelection

    masses = np.array([[0.4, 0.3, 0.3], [0.4, 0.3, 0.3]])
    selections = {
        "A": ThresholdSelection("A", 0.25, 0.0, 0.1, 100),
        "B": ThresholdSelection("B", 0.35, 0.0, 0.1, 100),
    }
    decided = apply_thresholds(masses, ["A", "B"], selections)
    assert list(decided) == [2, 0]


def test_affine_correction_is_linear_on_the_logit_scale() -> None:
    correction = AffineCorrection(slope=2.0, intercept=-1.0)
    assert correction.apply(np.array([0.0, 1.0])).tolist() == [-1.0, 1.0]
    assert correction.is_identity is False
    assert AffineCorrection(1.0, 0.0).is_identity


def test_site_calibrator_fits_transferred_and_leave_one_out_corrections() -> None:
    rng = np.random.default_rng(0)
    logits = np.concatenate([rng.normal(-1.0, 1.0, 60), rng.normal(1.0, 1.0, 60)])
    targets = (logits > 0.2).astype(float)
    sites = ["A"] * 60 + ["B"] * 60
    calibrator = SiteCalibrator.fit(logits, targets, sites, scope="site", external_sites=("D", "E"))
    assert set(calibrator.corrections) == {"A", "B"}
    assert set(calibrator.leave_one_out) == {"A", "B"}
    assert not calibrator.transferred.is_identity
    assert calibrator.correction_for("D").slope == pytest.approx(calibrator.transferred.slope)
    payload = calibrator.as_dict()
    assert payload["scope"] == "site" and "transferred" in payload


def test_site_calibrator_keeps_the_identity_when_a_site_has_one_class() -> None:
    logits = np.linspace(-1, 1, 20)
    targets = np.ones(20)
    calibrator = SiteCalibrator.fit(logits, targets, ["A"] * 20, scope="site", external_sites=())
    assert calibrator.corrections["A"].is_identity


def test_reference_categories_are_the_ordinal_positions() -> None:
    stages = [ALL_TRIPLES[0], ALL_TRIPLES[1], ALL_TRIPLES[31]]
    values = reference_categories(stages)
    assert values[0] == 0
    assert values[2] == 2


def test_stagefm_forward_shapes_and_projection(smoke_pipeline, smoke_batch) -> None:  # type: ignore[no-untyped-def]
    output = smoke_pipeline.model(smoke_batch)
    assert output.axis["T"].probabilities.shape[1] == 4
    assert output.axis["N"].probabilities.shape[1] == 4
    assert output.axis["M"].probabilities.shape[1] == 2
    assert output.projected.shape[-1] == 32
    assert torch.allclose(output.projected.sum(dim=-1), torch.ones(output.projected.shape[0]), atol=1e-5)
    assert torch.all(output.projected[:, ~smoke_pipeline.achievable.mask] == 0)


def test_stagefm_ablation_switches_change_the_support(smoke_config, smoke_batch) -> None:  # type: ignore[no-untyped-def]
    from dataclasses import replace

    from stagefm.cli.pipeline import prepare
    from stagefm.data.streams import StreamPlan

    no_consistency = replace(smoke_config, ablations={"stage_consistency": True})
    pipeline = prepare(no_consistency, "stagefm")
    plan = StreamPlan.from_config(no_consistency)
    assert plan.includes(Stream.CT)
    output = pipeline.model(smoke_batch)
    feasible_mask = torch.tensor(np.asarray(pipeline.achievable.mask, dtype=bool))
    assert bool((output.joint[:, ~feasible_mask] > 0).any())


def test_variant_flags_read_every_switch() -> None:
    flags = VariantFlags(stage_consistency=False, risk_control=False, fusion=False, ordinal=False, adaptation=False, post_hoc_projection=True)
    assert not flags.stage_consistency and flags.post_hoc_projection
    from_config = VariantFlags.from_config(ExperimentConfig.from_mapping(resolve_experiment("_smoke", "configs", ["experiment.ablations={'ct': true}"])))
    assert not from_config.use_image_tokens


def test_stream_only_model_uses_no_encoder(smoke_pipeline, smoke_batch) -> None:  # type: ignore[no-untyped-def]
    dimensions = smoke_pipeline.dimensions
    model = StreamOnlyModel(smoke_pipeline.config, ("radiomics", "clinical"), dimensions, smoke_pipeline.achievable)
    output = model(smoke_batch)
    assert output.projected.shape[-1] == 32
    assert model.components()["encoder"] == 0
    assert model.adapter_report()["adapter_fraction"] == 0.0
    assert model.monotonicity_violations() == 0


def test_arm_registry_covers_computed_and_quoted_arms() -> None:
    registry = arm_registry()
    assert "stagefm" in registry
    assert "finetuned_independent_heads" in registry
    assert set(QUOTED_ARMS) == {"eus", "reader_panel", "reader_panel_assisted", "published_multicentre_range"}
    rows = arm_table_rows()
    sources = {row["source"] for row in rows}
    assert {"computed", "quoted"} <= sources


def test_build_arm_constructs_every_trainable_arm(smoke_pipeline) -> None:  # type: ignore[no-untyped-def]
    dimensions = smoke_pipeline.dimensions
    plan = StreamPlan(active=(Stream.CT, Stream.RADIOMICS, Stream.CLINICAL, Stream.ENDOSCOPY, Stream.PATHOLOGY))
    for key, spec in BASELINES.items():
        model = build_arm(key, smoke_pipeline.config, plan, smoke_pipeline.achievable, dimensions)
        assert isinstance(model, torch.nn.Module), key
        assert spec.label
    with pytest.raises(KeyError):
        build_arm("does_not_exist", smoke_pipeline.config, plan, smoke_pipeline.achievable, dimensions)


def test_label_matrix_matches_the_triple_enumeration() -> None:
    matrix = label_matrix()
    assert matrix.shape == (32, 3)
    assert matrix[0].tolist() == [1, 0, 0]
    assert matrix[31].tolist() == [4, 3, 1]
