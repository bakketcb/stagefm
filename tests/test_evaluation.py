"""Evaluation tests: prediction bundles, per-site and subgroup analysis, reporting."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from stagefm.data.cohort import Layer
from stagefm.data.schema import StageTriple
from stagefm.data.staging import AchievableSet
from stagefm.evaluation.ascertainment import (
    ascertainment_report,
    bootstrap_report,
    nodal_error_indicator,
    tumour_error_indicator,
    vendor_groups_from,
)
from stagefm.evaluation.in_vitro import (
    BIOLOGICAL_REPLICATES,
    blinded_remeasurement_agreement,
    blocked_condition_effect,
    conditions,
    design_table,
    in_vitro_report,
    knockdown_reversal,
    ratio_to_control,
    texture_density_correlation,
)
from stagefm.evaluation.loop import PredictionBundle, category_mass_table, collect_predictions
from stagefm.evaluation.per_site import nodal_error_by_site, per_site_report, pooled_summary, tumour_error_by_site
from stagefm.evaluation.reader_study import (
    ReaderStudyDesign,
    ReaderTable,
    arm_summary,
    paired_reader_difference,
    reader_performance,
    reader_site_breakdown,
    reader_study_report,
)
from stagefm.evaluation.report import (
    ablation_row,
    arm_row,
    clinical_relevance,
    interaction_section,
    parity_check,
    prespecified_criterion,
    quoted_row,
    results_table,
)
from stagefm.evaluation.sensitivity import boundary_shift, consensus_exclusion, sensitivity_report
from stagefm.evaluation.silent_mode import (
    DeploymentLog,
    deployment_report,
    reading_time_distribution,
    stratified_open_rate,
    summarise_deployment,
    synthetic_deployment_log,
    timing_composition,
)
from stagefm.evaluation.subgroups import subgroup_columns, subgroup_report
from stagefm.training.amp import PrecisionSpec


@pytest.fixture(scope="module")
def external_bundle(request):  # type: ignore[no-untyped-def]
    """Predictions on the smoke external layer, built once for the module."""
    pipeline = request.getfixturevalue("smoke_pipeline")
    return collect_predictions(pipeline.model, pipeline.loader(Layer.EXTERNAL), PrecisionSpec.resolve("fp32", "cpu"), torch.device("cpu"))


def test_prediction_bundle_decodes_and_subsets(external_bundle) -> None:
    assert len(external_bundle) > 0
    assert external_bundle.predicted_columns.shape == (len(external_bundle),)
    assert np.all(external_bundle.predicted_columns >= 0)
    assert np.all(external_bundle.predicted_columns < 32)
    mask = np.zeros(len(external_bundle), dtype=bool)
    mask[0] = True
    subset = external_bundle.subset(mask)
    assert len(subset) == 1
    assert subset.sites == [external_bundle.sites[0]]
    assert subset.record_ids == [external_bundle.record_ids[0]]
    assert len(external_bundle.rows()) == len(external_bundle)
    state = external_bundle.as_state()
    assert state["projected"].shape == external_bundle.projected.shape


def test_prediction_bundle_rejects_a_wrong_selector(external_bundle) -> None:
    with pytest.raises(ValueError):
        external_bundle.subset(np.zeros(3, dtype=bool))


def test_category_mass_table_sums_to_one(external_bundle) -> None:
    masses = category_mass_table(external_bundle)
    assert np.allclose(masses.sum(axis=1), 1.0, atol=1e-5)


def test_per_site_report_covers_every_site(external_bundle, smoke_pipeline) -> None:  # type: ignore[no-untyped-def]
    report = per_site_report(external_bundle, smoke_pipeline.achievable, smoke_pipeline.config.cohort.regions)
    assert {row.site for row in report.rows} == set(external_bundle.sites)
    assert report.pooled["count"] == float(len(external_bundle))
    spread = report.spread("t_auroc")
    assert spread["spread"] >= 0.0
    payload = report.as_dict()
    assert "rows" in payload and "pooled" in payload and "t_auroc_spread" in payload


def test_pooled_summary_matches_the_bundle(external_bundle, smoke_pipeline) -> None:  # type: ignore[no-untyped-def]
    summary = pooled_summary(external_bundle, smoke_pipeline.achievable)
    assert 0.0 <= summary["concordance"] <= 1.0
    assert 0.0 <= summary["feasibility_percent"] <= 100.0
    assert summary["discordance_percent"] >= 0.0
    assert 0.0 <= summary["expected_calibration_error"] <= 1.0


def test_per_site_errors_are_one_minus_the_area(external_bundle) -> None:
    nodal = nodal_error_by_site(external_bundle)
    tumour = tumour_error_by_site(external_bundle)
    assert set(nodal) == set(tumour) == set(external_bundle.sites)
    assert all(0.0 <= value <= 1.0 for value in nodal.values())
    assert all(0.0 <= value <= 1.0 for value in tumour.values())


def test_subgroup_report_has_every_family(external_bundle, smoke_pipeline) -> None:  # type: ignore[no-untyped-def]
    records = smoke_pipeline.records(Layer.EXTERNAL)
    columns = subgroup_columns(records, smoke_pipeline.config.cohort.adequate_node_yield)
    report = subgroup_report(external_bundle, columns, smoke_pipeline.achievable)
    families = {row["family"] for row in report["rows"]}
    assert {"nodal_sampling", "lauren", "neoadjuvant", "sex", "age_band", "scanner_vendor"} <= families
    assert report["null_bands"]["n_auroc"] == pytest.approx(0.02)
    assert report["null_bands"]["discordance_points"] == pytest.approx(3.0)
    for summary in report["families"]:
        assert summary["strata"] >= 1


def test_ascertainment_report_structure(external_bundle, smoke_pipeline) -> None:  # type: ignore[no-untyped-def]
    columns = subgroup_columns(smoke_pipeline.records(Layer.EXTERNAL), smoke_pipeline.config.cohort.adequate_node_yield)
    report = ascertainment_report(
        external_bundle,
        columns,
        smoke_pipeline.config.cohort.median_harvested_nodes,
        smoke_pipeline.config.cohort.adequate_node_yield,
    )
    payload = report.as_dict()
    assert set(payload["nodal_error"]) == set(external_bundle.sites)
    assert payload["restricted_count"] <= len(external_bundle)
    assert 0.0 <= payload["retained_share"] <= 1.0
    nodal = payload["nodal_correlation"]
    assert "r" in nodal and "p" in nodal
    assert np.isfinite(payload["full_decomposition"]["between_site_variance"]) or len(external_bundle.sites) < 2
    indicator = nodal_error_indicator(external_bundle)
    assert set(np.unique(indicator)) <= {0.0, 1.0}
    tumour = tumour_error_indicator(external_bundle)
    assert tumour.shape == indicator.shape


def test_vendor_groups_partition_the_records(external_bundle, smoke_pipeline) -> None:  # type: ignore[no-untyped-def]
    columns = subgroup_columns(smoke_pipeline.records(Layer.EXTERNAL), 16)
    grouped = vendor_groups_from(columns)
    assert sum(len(positions) for positions in grouped.values()) == len(columns["scanner_vendor"])
    assert len(grouped) >= 1


def test_bootstrap_ascertainment_report_shapes(external_bundle) -> None:
    report = bootstrap_report(external_bundle, cutoff=16, replicates=20, seed=0)
    assert set(report) == {"full", "restricted"}
    assert "between_site_variance" in report["full"]


def test_sensitivity_analyses_include_their_own_control(external_bundle, smoke_pipeline) -> None:  # type: ignore[no-untyped-def]
    flags = np.array([record.consensus_staged for record in smoke_pipeline.records(Layer.EXTERNAL)], dtype=bool)
    report = sensitivity_report(external_bundle, flags)
    shifts = report["boundary_shift"]
    assert len(shifts) == 9
    control = next(item for item in shifts if item["name"] == "predicted_shift+0_reference_shift+0")
    from stagefm.metrics.decision import discordance_summary

    expected = discordance_summary(external_bundle.predicted_columns, external_bundle.stage_column, list(external_bundle.sites)).pooled
    assert control["discordance_percent"] == pytest.approx(expected)
    assert report["control_discordance_percent"] == pytest.approx(expected)
    assert "consensus_exclusion" in report
    excluded = consensus_exclusion(external_bundle, flags)
    assert excluded.count == int((~flags).sum())


def test_boundary_shift_recomputes_the_endpoint() -> None:
    surgery = StageTriple(1, 0, 0).flat_index
    systemic = StageTriple(4, 0, 1).flat_index
    bundle = _synthetic_bundle([surgery, systemic], [surgery, surgery])
    control = boundary_shift(bundle, offsets=(0,))
    assert control[0].discordance_percent == pytest.approx(50.0)
    assert control[0].name == "predicted_shift+0_reference_shift+0"
    both_up = boundary_shift(bundle, offsets=(1,))
    assert both_up[0].discordance_percent == pytest.approx(50.0)
    clamped = boundary_shift(bundle, offsets=(-1,))
    assert clamped[0].discordance_percent >= 0.0


def _synthetic_bundle(predicted: list[int], reference: list[int], sites: list[str] | None = None) -> PredictionBundle:
    count = len(predicted)
    projected = np.zeros((count, 32))
    projected[np.arange(count), np.array(predicted)] = 1.0
    labels = {"t": np.zeros(count, dtype=int), "n": np.zeros(count, dtype=int), "m": np.zeros(count, dtype=int)}
    return PredictionBundle(
        t_prob=np.full((count, 4), 0.25),
        n_prob=np.full((count, 4), 0.25),
        m_prob=np.full((count, 2), 0.5),
        joint=projected.copy(),
        projected=projected,
        boundary_logit=np.zeros(count),
        labels=labels,
        sites=sites or ["A"] * count,
        harvested_nodes=np.full(count, 20),
        record_ids=[f"r{index}" for index in range(count)],
        stage_column=np.array(reference, dtype=int),
    )


def test_reader_study_design_arithmetic() -> None:
    design = ReaderStudyDesign()
    assert design.paired_observations == 6480
    assert design.validate() == []
    assert design.as_dict()["expected_span"] == 360


def test_reader_study_pairs_and_reports() -> None:
    rng = np.random.default_rng(0)
    readers = 4
    exams = 6
    reader_id = np.repeat(np.arange(readers), exams * 2)
    exam_id = np.tile(np.arange(exams), readers * 2)
    assisted = np.tile(np.array([False] * exams + [True] * exams), readers)
    reference_t = np.tile(rng.integers(0, 4, exams), readers * 2)
    predicted_t = np.where(assisted, reference_t, (reference_t + 1) % 4)
    table = ReaderTable(
        reader_id=reader_id,
        exam_id=exam_id,
        site=np.tile(np.array(["A"] * exams + ["A"] * exams), readers),
        assisted=assisted,
        predicted_t=predicted_t,
        reference_t=reference_t,
        predicted_column=reference_t * 2,
        reference_column=reference_t * 2,
        elapsed_days=np.full(readers * exams * 2, 20.0),
    )
    assert table.validate_pairing(14) == []
    performance = reader_performance(table.arm(True))
    assert len(performance) == readers
    assert all(row.t_accuracy == pytest.approx(1.0) for row in performance)
    assert arm_summary(table.arm(False))["t_accuracy"] == pytest.approx(0.0)
    difference = paired_reader_difference(table.arm(False), table.arm(True), "accuracy", resamples=100, seed=0)
    assert difference["difference"] == pytest.approx(1.0)
    assert difference["low"] == pytest.approx(1.0)
    report = reader_study_report(table)
    assert report.as_dict()["design"]["readers"] == 18
    assert reader_site_breakdown(table.arm(True))
    with pytest.raises(ValueError):
        paired_reader_difference(table.arm(False), table.arm(True), "nonsense")
    shorter = ReaderTable(
        reader_id=table.reader_id[:2],
        exam_id=table.exam_id[:2],
        site=table.site[:2],
        assisted=table.assisted[:2],
        predicted_t=table.predicted_t[:2],
        reference_t=table.reference_t[:2],
        predicted_column=table.predicted_column[:2],
        reference_column=table.reference_column[:2],
    )
    with pytest.raises(ValueError):
        paired_reader_difference(table.arm(False), shorter, "accuracy")


def test_reader_study_flags_a_short_washout() -> None:
    table = ReaderTable(
        reader_id=np.array([0, 0, 1, 1]),
        exam_id=np.array([0, 0, 0, 0]),
        site=np.array(["A"] * 4),
        assisted=np.array([False, True, False, True]),
        predicted_t=np.array([0, 0, 1, 1]),
        reference_t=np.array([0, 1, 1, 0]),
        predicted_column=np.array([0, 0, 2, 2]),
        reference_column=np.array([0, 2, 2, 0]),
        elapsed_days=np.array([3.0, 3.0, 3.0, 3.0]),
    )
    problems = table.validate_pairing(14)
    assert any("washout" in problem for problem in problems)


def test_in_vitro_design_and_estimators() -> None:
    assert len(conditions()) == 6
    rows = design_table()
    assert len(rows) == 6 * BIOLOGICAL_REPLICATES * 2
    assert {row["condition"] for row in rows} >= {"unconditioned", "high_vegfc", "high_vegfc_sirna"}


def test_in_vitro_ratio_and_knockdown() -> None:
    condition = np.array(["unconditioned"] * 6 + ["high_vegfc"] * 6 + ["high_vegfc_sirna"] * 6)
    measurement = np.concatenate([np.ones(6), np.ones(6) * 2.0, np.ones(6) * 1.05])
    estimate = ratio_to_control(measurement, condition, "unconditioned", "high_vegfc", resamples=100, seed=0)
    assert estimate.ratio == pytest.approx(2.0)
    assert estimate.low == pytest.approx(2.0) and estimate.high == pytest.approx(2.0)
    reversal = knockdown_reversal(measurement, condition, "unconditioned", "high_vegfc", "high_vegfc_sirna")
    assert reversal["treated_ratio"] == pytest.approx(2.0)
    assert reversal["knockdown_ratio"] == pytest.approx(1.05)


def test_ratio_to_control_handles_a_missing_condition() -> None:
    condition = np.array(["unconditioned", "unconditioned"])
    measurement = np.array([1.0, 1.0])
    estimate = ratio_to_control(measurement, condition, "unconditioned", "absent_condition", resamples=10, seed=0)
    assert np.isnan(estimate.ratio)


def test_blocked_condition_effect_decomposes_the_variance() -> None:
    rng = np.random.default_rng(0)
    condition = np.repeat(["unconditioned", "high_vegfc", "low_vegfc"], 12)
    plate = np.tile(np.repeat([0, 1], 6), 3)
    measurement = np.repeat([1.0, 2.4, 1.3], 12) + 0.2 * plate + rng.normal(scale=0.05, size=36)
    decomposition = blocked_condition_effect(measurement, condition, plate)
    assert decomposition["ss_condition"] > decomposition["ss_residual"]
    assert decomposition["df_condition"] == 2.0
    assert decomposition["df_plate"] == 1.0


def test_texture_correlation_and_blinded_agreement() -> None:
    texture = np.linspace(0.0, 1.0, 40)
    density = texture * 2.0 + 0.05
    correlation = texture_density_correlation(texture, density, resamples=100, seed=0)
    assert correlation["rho"] == pytest.approx(1.0)
    agreement = blinded_remeasurement_agreement(texture, texture + 0.01)
    assert agreement["mean_difference"] == pytest.approx(0.01)
    assert np.isnan(blinded_remeasurement_agreement(np.array([1.0]), np.array([1.0]))["correlation"])


def test_in_vitro_report_is_empty_without_a_measurement_block() -> None:
    report = in_vitro_report()
    assert report.measured is False
    assert report.design_rows > 0
    assert report.as_dict()["tube_formation"] == {}


def test_in_vitro_report_with_a_block() -> None:
    rng = np.random.default_rng(1)
    condition = np.array(["unconditioned"] * 6 + ["high_vegfc"] * 6 + ["high_vegfc_sirna"] * 6)
    tube = np.concatenate([np.ones(6), np.ones(6) * 2.4, np.ones(6) * 1.08]) + rng.normal(scale=0.02, size=18)
    phosphorylation = np.concatenate([np.ones(6), np.ones(6) * 3.7, np.ones(6) * 1.2]) + rng.normal(scale=0.03, size=18)
    plate = np.tile([0, 1, 0, 1, 0, 1], 3)
    report = in_vitro_report(tube, phosphorylation, condition, plate, texture=np.linspace(0, 1, 30), vessel_density=np.linspace(0, 2, 30))
    assert report.measured
    assert report.tube_formation["treated_ratio"] == pytest.approx(2.4, abs=0.05)
    assert report.blocked_model["ss_condition"] > 0
    assert report.texture_correlation["rho"] == pytest.approx(1.0)


def test_deployment_summary_and_strata() -> None:
    log = DeploymentLog(
        site=np.array(["A", "A", "B", "B"], dtype=object),
        opened=np.array([True, False, True, True]),
        time_to_output_minutes=np.array([5.0, 7.0, 6.0, 9.0]),
        reading_time_seconds=np.array([5.0, 70.0, 15.0, 25.0]),
        unavailable=np.array([False, True, False, False]),
        model_minutes=np.array([4.0, 5.0, 4.5, 6.0]),
        preoperative_discordant=np.array([True, False, False, False]),
    )
    summary = summarise_deployment(log)
    assert summary.overall["eligible"] == 4.0
    assert summary.overall["opened"] == 3.0
    assert summary.per_site["A"]["opened_percent"] == pytest.approx(50.0)
    assert summary.per_site["B"]["opened_percent"] == pytest.approx(100.0)
    distribution = reading_time_distribution(log)
    assert distribution["under_10s_percent"] == pytest.approx(25.0)
    assert distribution["over_60s_percent"] == pytest.approx(25.0)
    strata = stratified_open_rate(log)
    assert strata["discordant"]["opened_percent"] == pytest.approx(100.0)
    timing = timing_composition(log)
    assert 0.0 < timing["model_share"] < 1.0
    assert timing["queue_and_transfer_minutes"] > 0
    report = deployment_report(log)
    assert set(report) == {"summary", "reading_time", "stratified_open_rate", "timing"}


def test_deployment_summary_without_optional_columns() -> None:
    log = DeploymentLog(
        site=np.array(["A"], dtype=object),
        opened=np.array([True]),
        time_to_output_minutes=np.array([6.0]),
        reading_time_seconds=np.array([18.0]),
        unavailable=np.array([False]),
    )
    assert stratified_open_rate(log) == {}
    assert timing_composition(log) == {}


def test_synthetic_deployment_log_reproduces_the_reported_counts() -> None:
    log = synthetic_deployment_log({"A": 612, "B": 498, "C": 402, "D": 338}, {"A": 0.846}, seed=0)
    assert len(log) == 1850
    summary = summarise_deployment(log)
    assert summary.overall["eligible"] == 1850.0
    assert set(summary.per_site) == {"A", "B", "C", "D"}


def test_arm_row_and_quoted_row_sources(external_bundle, smoke_pipeline) -> None:  # type: ignore[no-untyped-def]
    row = arm_row("stagefm", "STAGEFM", external_bundle, smoke_pipeline.achievable)
    assert row.source == "computed"
    assert row.as_dict()["arm"] == "stagefm"
    quoted = quoted_row("eus", "Endoscopic ultrasound")
    assert quoted.source == "quoted"
    assert quoted.discordance_percent == pytest.approx(11.82)
    assert quoted.detail["reference"]


def test_parity_criterion_and_clinical_relevance() -> None:
    model = arm_row_for(0.9445, 0.8214, 0.9032, 0.8312, 3.12)
    unconstrained = arm_row_for(0.9451, 0.8227, 0.9044, 0.7826, 7.44)
    parity = parity_check(model, unconstrained)
    assert parity["within_tolerance"] == 1.0
    criterion = prespecified_criterion(model.concordance, model.t_auroc)
    assert criterion["both_met"] is True
    relevance = clinical_relevance(3.12, 7.19)
    assert relevance["reduction_points"] == pytest.approx(4.07)
    assert relevance["exceeds_threshold"] == 1.0
    failing = prespecified_criterion(0.70, 0.85)
    assert failing["both_met"] is False


def arm_row_for(t: float, n: float, m: float, concordance: float, discordance: float):  # type: ignore[no-untyped-def]
    from stagefm.evaluation.report import ArmRow

    return ArmRow(
        arm="x",
        label="x",
        source="computed",
        t_auroc=t,
        n_auroc=n,
        m_auroc=m,
        concordance=concordance,
        weighted_kappa=0.8,
        discordance_percent=discordance,
        expected_calibration_error=0.02,
        feasibility_percent=100.0,
        count=3456,
    )


def test_ablation_row_and_interaction_section() -> None:
    achievable = AchievableSet()
    bundle = _synthetic_bundle([StageTriple(4, 0, 1).flat_index], [StageTriple(1, 0, 0).flat_index])
    row = ablation_row("without_stage_consistency", "Without I1", bundle, achievable)
    assert row.as_dict()["key"] == "without_stage_consistency"
    discordance_rows = {"full": 3.12, "without_stage_consistency": 5.87, "without_risk_control": 7.51, "without_both_structural": 8.92}
    concordance_rows = {"full": 0.8312, "without_stage_consistency": 0.8271, "without_risk_control": 0.8291, "without_both_structural": 0.8058}
    section = interaction_section(discordance_rows, concordance_rows)
    assert section["discordance"]["ratio"] == pytest.approx(1.3004, abs=1e-4)
    assert section["concordance"]["ratio"] == pytest.approx(0.5695, abs=1e-3)


def test_results_table_attaches_the_parity_comparison() -> None:
    rows = [
        arm_row_for(0.9445, 0.8214, 0.9032, 0.8312, 3.12),
        arm_row_for(0.9451, 0.8227, 0.9044, 0.7826, 7.44),
    ]
    rows[0] = dataclasses.replace(rows[0], arm="stagefm", label="STAGEFM")
    rows[1] = dataclasses.replace(rows[1], arm="finetuned_independent_heads")
    table = results_table(rows)
    assert table["parity"]["against"] == "finetuned_independent_heads"
    assert table["criterion"]["both_met"] is True
    assert len(table["rows"]) == 2
