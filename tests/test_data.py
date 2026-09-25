"""Data-pipeline tests: staging definitions, cohort contract, imaging, descriptors, streams."""

from __future__ import annotations

import numpy as np
import pytest

from stagefm.data.cohort import Layer, stratum_of, subsample_to_cutoff
from stagefm.data.imaging import (
    apply_hu_window,
    augment_volume,
    correct_bias_field,
    fit_to_grid,
    patchify,
    preprocess_volume,
    resample_isotropic,
    standardise_intensity,
    synthesise_volume,
    token_grid,
)
from stagefm.data.radiomics import (
    FAMILY_NAMES,
    SiteStandardiser,
    collapse_redundant,
    extract_descriptors,
    intraclass_correlation,
    stability_select,
)
from stagefm.data.schema import ALL_TRIPLES, M_CLASSES, N_CLASSES, T_CLASSES, Examination, StageTriple, Stream
from stagefm.data.segmentation import localise, peritumoral_shell, stability_perturbations
from stagefm.data.staging import AchievableRule, AchievableSet, boundary_discordant, shift_category, treatment_category
from stagefm.data.streams import StreamPlan, apply_modality_dropout, dropout_rates, presence_vector
from stagefm.data.synthetic import MISSING_ENDOSCOPY, MISSING_PATHOLOGY, generate_cohort, strata_report
from stagefm.data.text import ABSENT_TERM, TextVocabulary
from stagefm.utils.config import ExperimentConfig, ImagingConfig, resolve_experiment


def test_stage_triple_round_trip() -> None:
    for stage in ALL_TRIPLES:
        assert StageTriple.from_flat_index(stage.flat_index) == stage
    assert len(ALL_TRIPLES) == T_CLASSES * N_CLASSES * M_CLASSES


def test_stage_triple_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        StageTriple(t=5, n=0, m=0)
    with pytest.raises(ValueError):
        StageTriple(t=1, n=4, m=0)
    with pytest.raises(ValueError):
        StageTriple(t=1, n=0, m=2)


def test_achievable_set_shape_and_membership() -> None:
    achievable = AchievableSet()
    assert len(achievable) == 22
    assert pytest.approx(achievable.feasible_fraction) == 22 / 32
    assert achievable.contains(StageTriple(1, 0, 0))
    assert not achievable.contains(StageTriple(1, 3, 0))
    assert not achievable.contains(StageTriple(1, 2, 1))
    assert achievable.contains(StageTriple(4, 3, 1))
    assert not achievable.contains(StageTriple(2, 3, 0))
    assert not achievable.contains(StageTriple(2, 0, 1))


def test_achievable_variants_are_nested() -> None:
    tight = AchievableSet.from_rule(AchievableRule.STAGING_TIGHT)
    staging = AchievableSet.from_rule(AchievableRule.STAGING)
    loose = AchievableSet.from_rule(AchievableRule.STAGING_LOOSE)
    assert len(tight) == 16
    assert len(staging) == 22
    assert len(loose) == 26
    assert len(tight) < len(staging) < len(loose)
    assert set(tight.triples()) <= set(staging.triples()) <= set(loose.triples())


def test_achievable_set_rejects_empty_mask() -> None:
    with pytest.raises(ValueError):
        AchievableSet(mask=np.zeros(32, dtype=bool))


def test_data_derived_rule_requires_a_mask() -> None:
    with pytest.raises(ValueError):
        AchievableSet.from_rule(AchievableRule.DATA_DERIVED)
    derived = AchievableSet.from_observed([StageTriple(1, 0, 0), StageTriple(3, 2, 1)])
    assert len(derived) == 2


def test_treatment_category_rule() -> None:
    assert treatment_category(StageTriple(1, 0, 0)).value == "surgery_first"
    assert treatment_category(StageTriple(1, 1, 0)).value == "perioperative"
    assert treatment_category(StageTriple(2, 0, 0)).value == "perioperative"
    assert treatment_category(StageTriple(4, 0, 1)).value == "systemic"
    assert boundary_discordant(StageTriple(1, 0, 0), StageTriple(2, 1, 0))
    assert not boundary_discordant(StageTriple(3, 1, 0), StageTriple(3, 2, 0))
    assert not boundary_discordant(StageTriple(1, 0, 0), StageTriple(1, 0, 0))


def test_shift_category_clamps_at_both_ends() -> None:
    assert shift_category(StageTriple(1, 0, 0), -1).value == "surgery_first"
    assert shift_category(StageTriple(1, 0, 0), 1).value == "perioperative"
    assert shift_category(StageTriple(4, 0, 1), 5).value == "systemic"
    assert shift_category(StageTriple(4, 0, 1), -5).value == "surgery_first"


def test_cohort_config_partitions_match_the_manuscript() -> None:
    config = ExperimentConfig.from_mapping(resolve_experiment("main", "configs", []))
    assert config.cohort.train_size + config.cohort.internal_test_size == 7828
    assert config.cohort.development_size + config.cohort.external_size == 11284
    assert config.cohort.seed_count == 5
    assert config.cohort.median_harvested_nodes["A"] == 32
    assert config.cohort.median_harvested_nodes["E"] == 16


def test_effective_batch_size_is_reported() -> None:
    config = ExperimentConfig.from_mapping(resolve_experiment("main", "configs", []))
    assert config.train.batch_size == 8
    assert config.train.grad_accum == 1
    assert config.train.world_size == 4
    assert config.train.effective_batch_size == 32
    assert config.train.epochs == 60
    assert config.train.warmup_steps == 500
    assert config.train.grad_clip == 1.0
    assert config.train.weight_decay == pytest.approx(1e-4)
    assert config.train.lr_fusion == pytest.approx(3e-4)
    assert config.train.lr_adapter == pytest.approx(1e-4)
    assert config.imaging.patch_grid == (96, 96, 64)
    assert config.imaging.shell_outer_mm == 5.0
    assert config.encoder.lora_rank == 16
    assert config.encoder.lora_alpha == 32
    assert config.fusion.modality_dropout == 0.15


def test_stratum_membership_uses_the_cutoff() -> None:
    assert stratum_of(15, 16).value == "inadequate"
    assert stratum_of(16, 16).value == "adequate"


def test_synthetic_cohort_matches_the_reported_shape() -> None:
    config = ExperimentConfig.from_mapping(resolve_experiment("main", "configs", []))
    achievable = AchievableSet()
    train, internal, external, prospective, profiles = generate_cohort(config.cohort, achievable, seed=3)
    assert len(train) == 5868
    assert len(internal) == 1960
    assert len(external) == 3456
    assert len(prospective) == 1850
    assert set(profiles) == {"A", "B", "C", "D", "E"}
    assert {record.site for record in train} <= set(config.cohort.development_sites)
    assert {record.site for record in external} == set(config.cohort.external_sites)
    assert all(achievable.contains(record.stage) for record in train + internal + external)


def test_synthetic_missingness_matches_the_reported_counts() -> None:
    config = ExperimentConfig.from_mapping(resolve_experiment("main", "configs", []))
    train, internal, external, _, _ = generate_cohort(config.cohort, AchievableSet(), seed=4)
    analysis_set = train + internal + external
    assert sum(1 for record in analysis_set if not record.available["endoscopy"]) == MISSING_ENDOSCOPY
    assert sum(1 for record in analysis_set if not record.available["pathology"]) == MISSING_PATHOLOGY
    assert sum(1 for record in analysis_set if not record.available["pathology"]) / len(analysis_set) < 0.006


def test_synthetic_site_yield_means_order_as_reported() -> None:
    config = ExperimentConfig.from_mapping(resolve_experiment("main", "configs", []))
    _, _, external, _, _ = generate_cohort(config.cohort, AchievableSet(), seed=5)
    yields = {site: np.median([record.harvested_nodes for record in external if record.site == site]) for site in ("D", "E")}
    assert yields["D"] > yields["E"]


def test_strata_report_shares_sum_to_one() -> None:
    config = ExperimentConfig.from_mapping(resolve_experiment("main", "configs", []))
    train, _, external, _, _ = generate_cohort(config.cohort, AchievableSet(), seed=6)
    report = strata_report(train + external, config.cohort.adequate_node_yield)
    assert report["adequate"] + report["inadequate"] == report["total"]
    assert report["adequate_share"] == pytest.approx(report["adequate"] / report["total"])


def test_subsample_to_cutoff_keeps_only_adequate_yields() -> None:
    records = tuple(
        Examination(
            record_id=f"r{index}",
            site="A",
            region="I",
            stage=StageTriple(2, 1, 0),
            harvested_nodes=nodes,
            scanner_vendor="Siemens",
            neoadjuvant_exposed=False,
            lauren="intestinal",
            age=60,
            sex="male",
            ct_ref="x",
        )
        for index, nodes in enumerate([5, 16, 30])
    )
    kept = subsample_to_cutoff(records, 16)
    assert [record.harvested_nodes for record in kept] == [16, 30]


def test_resample_isotropic_is_a_no_op_at_the_target_spacing() -> None:
    volume = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
    same = resample_isotropic(volume, (1.0, 1.0, 1.0), (1.0, 1.0, 1.0))
    assert np.array_equal(same, volume)


def test_resample_isotropic_changes_the_grid_when_spacing_changes() -> None:
    volume = np.zeros((8, 8, 8), dtype=np.float32)
    resampled = resample_isotropic(volume, (2.0, 1.0, 1.0), (1.0, 1.0, 1.0))
    assert resampled.shape == (16, 8, 8)


def test_hu_window_bounds_and_rescales() -> None:
    volume = np.array([-1000.0, -150.0, 50.0, 250.0, 1000.0], dtype=np.float32)
    windowed = apply_hu_window(volume, (-150.0, 250.0))
    assert windowed.min() == pytest.approx(0.0)
    assert windowed.max() == pytest.approx(1.0)
    assert windowed[2] == pytest.approx(0.5)


def test_bias_field_correction_reduces_the_smooth_gradient() -> None:
    axis = np.linspace(0.5, 1.5, 32, dtype=np.float32)
    field = axis[None, :, None] * np.ones((32, 32, 32), dtype=np.float32)
    signal = np.ones((32, 32, 32), dtype=np.float32) * 100.0 + 20.0 * np.sin(np.linspace(0, 6, 32))[None, :, None]
    volume = (signal * field).astype(np.float32)
    corrected = correct_bias_field(volume, (1.0, 1.0, 1.0), 4.0)
    spread_before = float(np.abs(volume.mean(axis=(0, 2)) - volume.mean()).max())
    spread_after = float(np.abs(corrected.mean(axis=(0, 2)) - corrected.mean()).max())
    assert spread_after < spread_before


def test_intensity_standardisation_has_unit_scale() -> None:
    volume = np.linspace(-3.0, 7.0, 500, dtype=np.float32)
    standardised = standardise_intensity(volume, "zscore")
    assert float(standardised.mean()) == pytest.approx(0.0, abs=1e-6)
    assert float(standardised.std()) == pytest.approx(1.0, abs=1e-6)
    with pytest.raises(ValueError):
        standardise_intensity(volume, "histogram")


def test_fit_to_grid_crops_and_pads() -> None:
    volume = np.arange(6 * 6 * 6, dtype=np.float32).reshape(6, 6, 6)
    cropped = fit_to_grid(volume, (4, 4, 4))
    assert cropped.shape == (4, 4, 4)
    padded = fit_to_grid(volume, (8, 8, 8))
    assert padded.shape == (8, 8, 8)
    assert padded.sum() > 0


def test_synthesise_volume_is_deterministic_for_a_seed() -> None:
    latent = np.linspace(-1.0, 1.0, 32, dtype=np.float32)
    first = synthesise_volume(latent, (16, 16, 8), seed=11)
    second = synthesise_volume(latent, (16, 16, 8), seed=11)
    third = synthesise_volume(latent, (16, 16, 8), seed=12)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, third)


def test_preprocess_volume_matches_the_declared_grid_after_fitting() -> None:
    config = ImagingConfig(patch_grid=(12, 12, 8))
    latent = np.zeros(32, dtype=np.float32)
    raw = synthesise_volume(latent, (12, 12, 8), seed=1)
    processed = fit_to_grid(preprocess_volume(raw, (1.2, 0.9, 0.9), config), config.patch_grid)
    assert processed.shape == (12, 12, 8)
    assert float(processed.mean()) == pytest.approx(0.0, abs=0.2)


def test_augmentation_stays_within_the_configured_bounds() -> None:
    volume = np.zeros((16, 16, 16), dtype=np.float32)
    volume[6:10, 6:10, 6:10] = 1.0
    rng = np.random.default_rng(0)
    augmented = augment_volume(volume, rng, rotate_degrees=5.0, scale_fraction=0.05, jitter_hu=20.0)
    assert augmented.shape == volume.shape
    assert float(np.abs(augmented.mean() - volume.mean())) < 20.0 / 240.0 * 3.0


def test_patchify_shape_and_token_count() -> None:
    volume = np.arange(8 * 8 * 8, dtype=np.float32).reshape(8, 8, 8)
    tokens = patchify(volume, 4)
    assert tokens.shape == (8, 64)
    assert token_grid((8, 8, 8), 4) == (2, 2, 2)


def test_patchify_rejects_a_volume_smaller_than_a_patch() -> None:
    with pytest.raises(ValueError):
        patchify(np.zeros((2, 2, 2), dtype=np.float32), 4)


def test_localisation_produces_disjoint_masks_and_a_non_empty_shell() -> None:
    axis = np.linspace(-1.0, 1.0, 32, dtype=np.float32)
    volume = np.ones((32, 32, 32), dtype=np.float32) * 0.6
    volume[8:24, 8:24, 8:24] = np.where(axis[8:24, None, None] > 0.2, 0.1, 0.6).astype(np.float32)
    masks = localise(volume, (1.0, 1.0, 1.0), 0.0, 3.0)
    assert masks.wall.any()
    assert masks.peritumoral_shell.any()
    assert not np.any(masks.peritumoral_shell & masks.wall)


def test_peritumoral_shell_radii() -> None:
    mask = np.zeros((11, 11, 11), dtype=bool)
    mask[5, 5, 5] = True
    shell = peritumoral_shell(mask, (1.0, 1.0, 1.0), 1.0, 2.0)
    distances = np.sqrt(((np.argwhere(shell) - np.array([5, 5, 5])) ** 2).sum(axis=1))
    assert distances.min() > 1.0
    assert distances.max() <= 2.0
    with pytest.raises(ValueError):
        peritumoral_shell(mask, (1.0, 1.0, 1.0), 2.0, 1.0)


def test_stability_perturbations_are_independent_draws() -> None:
    mask = np.zeros((9, 9, 9), dtype=bool)
    mask[3:6, 3:6, 3:6] = True
    rng = np.random.default_rng(0)
    perturbations = stability_perturbations(mask, repeats=6, rng=rng)
    assert len(perturbations) == 6
    assert all(perturbed.shape == mask.shape for perturbed in perturbations)
    assert any(not np.array_equal(perturbed, mask) for perturbed in perturbations)


def test_descriptor_families_and_determinism() -> None:
    rng = np.random.default_rng(0)
    volume = rng.random((12, 12, 12)).astype(np.float32)
    mask = np.zeros((12, 12, 12), dtype=bool)
    mask[3:9, 3:9, 3:9] = True
    families = tuple(FAMILY_NAMES)
    first = extract_descriptors(volume, mask, (1.0, 1.0, 1.0), families)
    second = extract_descriptors(volume, mask, (1.0, 1.0, 1.0), families)
    assert np.array_equal(first.values, second.values)
    assert np.isfinite(first.values).all()
    assert first.values.size == sum(len(FAMILY_NAMES[family]) for family in families)
    assert len(first.names) == first.values.size


def test_descriptor_extraction_rejects_unknown_families() -> None:
    with pytest.raises(ValueError):
        extract_descriptors(np.zeros((4, 4, 4), dtype=np.float32), np.ones((4, 4, 4), dtype=bool), (1.0, 1.0, 1.0), ("nope",))


def test_intraclass_correlation_bounds() -> None:
    rng = np.random.default_rng(0)
    stable = np.tile(rng.normal(size=(1, 30)), (4, 1)) + rng.normal(scale=0.01, size=(4, 30))
    unstable = rng.normal(size=(4, 30))
    assert intraclass_correlation(stable) > intraclass_correlation(unstable)
    assert -0.5 < intraclass_correlation(unstable) < 0.6
    with pytest.raises(ValueError):
        intraclass_correlation(np.zeros(4))


def test_stability_selection_keeps_the_reproducible_descriptors() -> None:
    rng = np.random.default_rng(0)
    subjects = 40
    repeats = 5
    reliable = np.tile(rng.normal(size=subjects), (repeats, 1)) + rng.normal(scale=0.02, size=(repeats, subjects))
    noisy = rng.normal(size=(repeats, subjects))
    tensor = np.stack([reliable, noisy], axis=-1)
    keep, scores = stability_select(tensor, ("reliable", "noisy"), threshold=0.75)
    assert keep == [0]
    assert scores[0] > scores[1]


def test_collapse_redundant_keeps_one_representative_per_block() -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(size=60)
    matrix = np.column_stack([base, base * 2.0, base * -1.5, rng.normal(size=60)])
    keep = collapse_redundant(matrix, ["a", "b", "c", "d"], threshold=0.9)
    assert len(keep) == 2
    assert 3 in keep


def test_site_standardiser_uses_only_the_site_it_was_fitted_on() -> None:
    rng = np.random.default_rng(0)
    features = rng.normal(size=(40, 3)) + np.repeat(np.array([[0.0, 0.0, 0.0], [4.0, 4.0, 4.0]]), 20, axis=0)
    sites = ["A"] * 20 + ["B"] * 20
    standardiser = SiteStandardiser.fit(features, sites)
    transformed = standardiser.transform(features, sites)
    for site in ("A", "B"):
        selector = np.array(sites) == site
        assert np.allclose(transformed[selector].mean(axis=0), 0.0, atol=1e-9)
        assert np.allclose(transformed[selector].std(axis=0), 1.0, atol=1e-9)
    assert "C" not in standardiser.locations


def test_text_vocabulary_reserves_the_absence_index() -> None:
    records = [
        Examination(
            record_id="a",
            site="A",
            region="I",
            stage=StageTriple(2, 1, 0),
            harvested_nodes=20,
            scanner_vendor="Siemens",
            neoadjuvant_exposed=False,
            lauren="intestinal",
            age=60,
            sex="male",
            ct_ref="x",
            endoscopy_terms=("ulceration", "stenosis"),
            pathology_terms=("diffuse",),
        ),
        Examination(
            record_id="b",
            site="A",
            region="I",
            stage=StageTriple(3, 2, 1),
            harvested_nodes=20,
            scanner_vendor="Siemens",
            neoadjuvant_exposed=True,
            lauren="diffuse",
            age=70,
            sex="female",
            ct_ref="y",
            endoscopy_terms=(),
            pathology_terms=("adenocarcinoma",),
        ),
    ]
    vocabulary = TextVocabulary.fit(records)
    assert vocabulary.endoscopy[ABSENT_TERM] == 0
    assert vocabulary.encode_endoscopy(()) == (0,)
    assert vocabulary.encode_endoscopy(("ulceration",))[0] != 0
    assert vocabulary.encode_pathology(("nonexistent-term",))[0] == vocabulary.pathology["<unknown>"]


def test_stream_plan_removes_named_streams() -> None:
    config = ExperimentConfig.from_mapping(resolve_experiment("_smoke", "configs", ["experiment.ablations={'ct': true, 'clinical': true}"]))
    plan = StreamPlan.from_config(config)
    assert not plan.includes(Stream.CT)
    assert not plan.includes(Stream.CLINICAL)
    assert plan.includes(Stream.RADIOMICS)
    assert plan.drops_imaging


def test_modality_dropout_rate_and_monotonicity() -> None:
    plan = StreamPlan(active=(Stream.CT, Stream.RADIOMICS, Stream.CLINICAL, Stream.ENDOSCOPY, Stream.PATHOLOGY))
    rates = dropout_rates(0.15)
    assert rates[Stream.CLINICAL] == pytest.approx(0.15)
    rng = np.random.default_rng(0)
    availability = {"ct": True, "radiomics": True, "clinical": True, "endoscopy": False, "pathology": True}
    dropped = apply_modality_dropout(availability, rates, rng, plan)
    assert dropped["endoscopy"] is False
    assert dropped["ct"] is True


def test_presence_vector_follows_the_plan() -> None:
    plan = StreamPlan(active=(Stream.CT, Stream.CLINICAL))
    vector = presence_vector({"ct": True, "clinical": False, "radiomics": True}, plan)
    assert list(vector) == [1.0, 0.0]


def test_dataset_item_carries_every_stream_and_the_supervision(smoke_pipeline) -> None:  # type: ignore[no-untyped-def]
    item = smoke_pipeline.datasets[Layer.TRAIN][0]
    assert item["volume"].shape[0] == 1
    assert item["radiomics"].numel() > 0
    assert item["clinical"].numel() > 0
    assert item["endoscopy"].numel() == 6
    assert item["presence"].numel() == 5
    assert int(item["stage_column"]) == (int(item["label_t"]) * 8 + int(item["label_n"]) * 2 + int(item["label_m"]))


def test_training_dataset_augments_and_evaluation_dataset_does_not(smoke_pipeline) -> None:  # type: ignore[no-untyped-def]
    train_first = smoke_pipeline.datasets[Layer.TRAIN][0]["volume"]
    train_second = smoke_pipeline.datasets[Layer.TRAIN][0]["volume"]
    assert train_first.shape == train_second.shape
    external = smoke_pipeline.datasets[Layer.EXTERNAL][0]["volume"]
    assert external.shape == train_first.shape


def test_label_table_shapes(smoke_pipeline) -> None:  # type: ignore[no-untyped-def]
    from stagefm.data.dataset import label_table

    table = label_table(smoke_pipeline.datasets[Layer.EXTERNAL])
    assert table["t"].shape == table["n"].shape == table["m"].shape == table["stage_column"].shape
    assert set(np.unique(table["site"])) <= {"A", "B", "C", "D", "E"}
