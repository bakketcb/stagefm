"""Loss and metric tests, with expectations derived independently of the implementation."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from sklearn.metrics import cohen_kappa_score, roc_auc_score

from stagefm.data.schema import ALL_TRIPLES, StageTriple, TreatmentCategory
from stagefm.data.staging import treatment_category
from stagefm.losses.calibration import brier_surrogate, calibration_loss, site_gap_loss
from stagefm.losses.decision import decision_loss, discordance_surrogate, systemic_mass
from stagefm.losses.ordinal import class_probability_loss, cumulative_link_loss, ordinal_loss
from stagefm.losses.total import staged_objective
from stagefm.metrics.calibration import calibration_slope, calibration_summary, expected_calibration_error
from stagefm.metrics.concordance import axis_agreement, exact_combination_concordance, ordinal_distance_agreement, weighted_kappa
from stagefm.metrics.decision import (
    binary_nri,
    boundary_discordance,
    decision_curve,
    discordance_indicator,
    discordance_summary,
    net_benefit,
    net_reclassification_improvement,
)
from stagefm.metrics.discrimination import axis_discrimination, binary_auroc, expected_axis_score, macro_auroc, ordinal_auroc, ordinal_auroc_ci
from stagefm.metrics.feasibility import feasibility_breakdown, feasibility_rate, infeasible_examples
from stagefm.models.ordinal import OrdinalOutput, StagingHeads
from stagefm.utils.config import LossConfig, OrdinalConfig


def _constant_output(probabilities: torch.Tensor) -> OrdinalOutput:
    cut_points = torch.linspace(0.1, 0.9, probabilities.shape[1] - 1)
    return OrdinalOutput(
        probabilities=probabilities,
        cumulative_logits=torch.zeros(probabilities.shape[0], probabilities.shape[1] - 1),
        score=torch.zeros(probabilities.shape[0]),
        cut_points=cut_points,
    )


def test_cumulative_link_loss_matches_the_hand_computation() -> None:
    logits = torch.tensor([[1.5, -0.5], [-1.0, 2.0]], dtype=torch.float64)
    targets = torch.tensor([2, 1], dtype=torch.long)
    terms = []
    for row, target in enumerate(targets.tolist()):
        for level in range(logits.shape[1]):
            label = 1.0 if target > level else 0.0
            value = float(logits[row, level])
            terms.append(label * math.log1p(math.exp(-value)) + (1 - label) * math.log1p(math.exp(value)))
    assert float(cumulative_link_loss(logits, targets)) == pytest.approx(float(np.mean(terms)))


def test_cumulative_link_loss_has_no_terms_for_a_binary_axis_with_one_threshold() -> None:
    logits = torch.tensor([[0.5], [-0.5]], dtype=torch.float64)
    targets = torch.tensor([1, 0], dtype=torch.long)
    value = float(cumulative_link_loss(logits, targets))
    expected = math.log1p(math.exp(-0.5))
    assert value == pytest.approx(expected)


def test_class_probability_loss_matches_negative_log_likelihood() -> None:
    probabilities = torch.tensor([[0.7, 0.3], [0.2, 0.8]], dtype=torch.float64)
    targets = torch.tensor([0, 1], dtype=torch.long)
    expected = -(math.log(0.7) + math.log(0.8)) / 2
    assert float(class_probability_loss(probabilities, targets)) == pytest.approx(expected)


def test_ordinal_loss_uses_the_cumulative_form_only_for_the_ordinal_head() -> None:
    probabilities = torch.softmax(torch.randn(4, 3), dim=-1)
    axis = {
        "T": _constant_output(probabilities),
        "N": _constant_output(probabilities),
        "M": _constant_output(torch.softmax(torch.randn(4, 2), dim=-1)),
    }
    targets = {"T": torch.tensor([0, 1, 2, 1]), "N": torch.tensor([0, 1, 2, 1]), "M": torch.tensor([0, 1, 1, 0])}
    cumulative = ordinal_loss(axis, targets, ordinal=True)
    categorical = ordinal_loss(axis, targets, ordinal=False)
    assert cumulative.total.ndim == 0 and categorical.total.ndim == 0
    assert float(cumulative.total) != pytest.approx(float(categorical.total))


def test_systemic_mass_and_decision_loss_follow_the_projection() -> None:
    mapping = np.array([list(TreatmentCategory).index(treatment_category(stage)) for stage in ALL_TRIPLES])
    probabilities = np.zeros((2, 32), dtype=np.float64)
    probabilities[0, np.flatnonzero(mapping == 2)[0]] = 1.0
    probabilities[1, np.flatnonzero(mapping == 1)[0]] = 1.0
    tensor = torch.tensor(probabilities)
    mass = systemic_mass(tensor)
    assert float(mass[0]) == pytest.approx(1.0)
    assert float(mass[1]) == pytest.approx(0.0)
    reference = torch.tensor([np.flatnonzero(mapping == 2)[0], np.flatnonzero(mapping == 1)[0]])
    loss = decision_loss(tensor, reference)
    assert float(loss.total) < 1e-6
    swapped = decision_loss(tensor, reference.flip(0))
    assert float(swapped.total) > 1.0


def test_discordance_surrogate_is_the_soft_error_rate() -> None:
    mapping = np.array([list(TreatmentCategory).index(treatment_category(stage)) for stage in ALL_TRIPLES])
    probabilities = np.zeros((2, 32), dtype=np.float64)
    probabilities[0, np.flatnonzero(mapping == 2)[0]] = 1.0
    probabilities[1, np.flatnonzero(mapping == 0)[0]] = 1.0
    reference = torch.tensor([np.flatnonzero(mapping == 2)[0], np.flatnonzero(mapping == 2)[0]])
    value = float(discordance_surrogate(torch.tensor(probabilities), reference))
    assert value == pytest.approx(0.5)


def test_site_gap_loss_penalises_a_miscalibrated_site() -> None:
    probability = torch.tensor([0.9, 0.9, 0.1, 0.1])
    targets = torch.tensor([0.0, 0.0, 1.0, 1.0])
    sites = torch.tensor([0, 0, 1, 1])
    loss = site_gap_loss(probability, targets, sites)
    assert float(loss.total) == pytest.approx(0.9)
    assert set(loss.per_site_gap) == {0, 1}


def test_brier_surrogate_matches_the_mean_squared_error() -> None:
    probability = torch.tensor([0.3, 0.8])
    targets = torch.tensor([0.0, 1.0])
    assert float(brier_surrogate(probability, targets)) == pytest.approx((0.09 + 0.04) / 2)


def test_calibration_loss_variants() -> None:
    mapping = np.array([list(TreatmentCategory).index(treatment_category(stage)) for stage in ALL_TRIPLES])
    probabilities = np.zeros((2, 32), dtype=np.float64)
    probabilities[0, np.flatnonzero(mapping == 2)[0]] = 1.0
    probabilities[1, np.flatnonzero(mapping == 0)[0]] = 1.0
    reference = torch.tensor([np.flatnonzero(mapping == 2)[0], np.flatnonzero(mapping == 2)[0]])
    sites = torch.tensor([0, 1])
    gap = calibration_loss(torch.tensor(probabilities), reference, sites, variant="site_gap")
    brier = calibration_loss(torch.tensor(probabilities), reference, sites, variant="brier")
    assert float(gap.total) == pytest.approx(0.5)
    assert float(brier.total) == pytest.approx(0.5)


def test_staged_objective_is_the_documented_weighted_sum(smoke_pipeline, smoke_batch) -> None:  # type: ignore[no-untyped-def]
    model = smoke_pipeline.model
    output = model(smoke_batch)
    config = LossConfig(lambda_decision=0.5, gamma_calibration=0.1)
    breakdown = staged_objective(
        axis_outputs=output.axis,
        projected=output.projected,
        reference_stages=smoke_batch["stage_column"],
        targets={"T": smoke_batch["label_t"], "N": smoke_batch["label_n"], "M": smoke_batch["label_m"]},
        site_index=smoke_batch["site_index"],
        config=config,
        ordinal=True,
    )
    expected = breakdown.ordinal.total + 0.5 * breakdown.decision.total + 0.1 * breakdown.calibration.total
    assert float(breakdown.total.detach()) == pytest.approx(float(expected.detach()))
    floats = breakdown.as_floats()
    assert "loss/ordinal" in floats and "loss/decision" in floats and "loss/calibration" in floats


def test_binary_auroc_matches_pairwise_counting_and_the_library() -> None:
    positive = np.array([0.9, 0.6, 0.8, 0.35])
    negative = np.array([0.5, 0.2, 0.1])
    scores = np.concatenate([positive, negative])
    labels = np.concatenate([np.ones(positive.size), np.zeros(negative.size)])
    count = sum(float((value > negative).sum()) + 0.5 * float((value == negative).sum()) for value in positive) / (positive.size * negative.size)
    assert binary_auroc(scores, labels) == pytest.approx(count)
    assert binary_auroc(scores, labels) == pytest.approx(float(roc_auc_score(labels, scores)))


def test_macro_auroc_is_the_mean_of_one_versus_rest() -> None:
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 4, 200)
    scores = rng.random((200, 4)) + 0.7 * np.eye(4)[labels]
    expected = np.mean([roc_auc_score((labels == category).astype(int), scores[:, category]) for category in range(4)])
    assert macro_auroc(scores, labels, 4) == pytest.approx(float(expected))


def test_ordinal_auroc_reduces_to_the_binary_area_with_two_categories() -> None:
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 2, 150)
    scores = labels + rng.normal(scale=1.0, size=150)
    assert ordinal_auroc(scores, labels) == pytest.approx(binary_auroc(scores, labels))


def test_ordinal_auroc_is_a_weighted_mean_of_pairwise_areas() -> None:
    rng = np.random.default_rng(2)
    labels = rng.integers(0, 4, 240)
    scores = labels + rng.normal(scale=1.5, size=240)
    numerator = 0.0
    denominator = 0.0
    for lower in range(4):
        for higher in range(lower + 1, 4):
            low = scores[labels == lower]
            high = scores[labels == higher]
            if low.size == 0 or high.size == 0:
                continue
            pairwise = binary_auroc(np.concatenate([low, high]), np.concatenate([np.zeros(low.size), np.ones(high.size)]))
            numerator += pairwise * low.size * high.size
            denominator += low.size * high.size
    assert ordinal_auroc(scores, labels) == pytest.approx(numerator / denominator)


def test_axis_discrimination_returns_all_three_axes() -> None:
    rng = np.random.default_rng(3)
    labels = {
        "t": rng.integers(0, 4, 200),
        "n": rng.integers(0, 4, 200),
        "m": rng.integers(0, 2, 200),
    }
    t_scores = rng.random((200, 4)) + 0.6 * np.eye(4)[labels["t"]]
    n_scores = labels["n"] + rng.normal(scale=1.2, size=200)
    m_scores = labels["m"] + rng.normal(scale=1.0, size=200)
    report = axis_discrimination(t_scores, n_scores, m_scores, labels)
    assert set(report) == {"T", "N", "M"}
    assert report["T"].low <= report["T"].auroc <= report["T"].high


def test_ordinal_auroc_ci_brackets_the_estimate() -> None:
    rng = np.random.default_rng(4)
    labels = rng.integers(0, 4, 120)
    scores = labels + rng.normal(scale=1.0, size=120)
    result = ordinal_auroc_ci(scores, labels, resamples=200, seed=0)
    assert result.low <= result.auc <= result.high


def test_expected_axis_score_shapes() -> None:
    probabilities = np.array([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])
    assert expected_axis_score("T", probabilities).shape == (2, 4)
    assert expected_axis_score("N", probabilities).shape == (2,)
    assert expected_axis_score("M", np.array([[0.3, 0.7]])).tolist() == pytest.approx([0.7])


def test_exact_concordance_and_axis_agreement_are_hand_checkable() -> None:
    predicted = np.array([0, 1, 2, 3])
    reference = np.array([0, 1, 3, 3])
    assert exact_combination_concordance(predicted, reference) == pytest.approx(0.75)
    assert axis_agreement(predicted, reference) == pytest.approx(0.75)
    assert math.isnan(exact_combination_concordance(np.zeros(0, dtype=int), np.zeros(0, dtype=int)))


def test_weighted_kappa_matches_the_library_and_is_one_on_agreement() -> None:
    reference = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    predicted = np.array([0, 1, 1, 1, 2, 3, 3, 3])
    assert weighted_kappa(predicted, reference, "linear") == pytest.approx(float(cohen_kappa_score(reference, predicted, weights="linear")))
    assert weighted_kappa(reference, reference, "linear") == pytest.approx(1.0)
    quadratic = weighted_kappa(predicted, reference, "quadratic")
    assert quadratic != pytest.approx(weighted_kappa(predicted, reference, "linear"))


def test_ordinal_distance_agreement_bounds() -> None:
    reference = np.array([0, 5, 16], dtype=int)
    assert ordinal_distance_agreement(reference, reference) == pytest.approx(1.0)
    far = np.array([31, 31, 31], dtype=int)
    assert 0.0 <= ordinal_distance_agreement(far, reference) < 1.0


def test_expected_calibration_error_hand_example() -> None:
    probabilities = np.concatenate([np.full(80, 0.25), np.full(20, 0.85)])
    outcomes = np.concatenate([np.zeros(80), np.ones(20)])
    value = expected_calibration_error(probabilities, outcomes, bins=10)
    assert value == pytest.approx(0.8 * abs(0.25 - 0.0) + 0.2 * abs(0.85 - 1.0))


def test_calibration_summary_is_record_weighted() -> None:
    probabilities = np.concatenate([np.full(90, 0.5), np.full(10, 0.1)])
    outcomes = np.zeros(100)
    sites = ["A"] * 90 + ["B"] * 10
    summary = calibration_summary(probabilities, outcomes, sites, bins=10)
    expected = 0.9 * abs(0.5 - 0.0) + 0.1 * abs(0.1 - 0.0)
    assert summary.pooled == pytest.approx(expected)
    assert summary.counts == {"A": 90, "B": 10}
    assert summary.as_dict()["bins"] == 10


def test_calibration_slope_is_about_one_for_calibrated_predictions() -> None:
    rng = np.random.default_rng(5)
    logits = rng.normal(size=2000)
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    outcomes = (rng.random(2000) < probabilities).astype(float)
    assert calibration_slope(probabilities, outcomes) == pytest.approx(1.0, abs=0.25)


def test_net_benefit_closed_form_and_threshold_guard() -> None:
    positive = np.array([1, 1, 0, 0, 1, 0])
    predicted = np.array([1, 0, 1, 0, 1, 0])
    threshold = 0.3
    expected = ((predicted == 1) & (positive == 1)).sum() / 6 - ((predicted == 1) & (positive == 0)).sum() / 6 * (threshold / (1 - threshold))
    assert net_benefit(positive, predicted, threshold) == pytest.approx(float(expected))
    with pytest.raises(ValueError):
        net_benefit(positive, predicted, 0.0)


def test_decision_curve_reference_arms() -> None:
    positive = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    probabilities = np.array([0.9, 0.2, 0.7, 0.6, 0.3, 0.1, 0.8, 0.4])
    thresholds = np.array([0.2, 0.5])
    curve = decision_curve(positive, probabilities, thresholds)
    prevalence = float(positive.mean())
    assert np.allclose(curve["treat_none"], 0.0)
    assert np.allclose(curve["treat_all"], [prevalence - (1 - prevalence) * (t / (1 - t)) for t in thresholds])


def test_boundary_discordance_and_indicator() -> None:
    surgery = StageTriple(1, 0, 0).flat_index
    perioperative = StageTriple(2, 1, 0).flat_index
    systemic = StageTriple(4, 0, 1).flat_index
    predicted = np.array([surgery, perioperative, systemic])
    reference = np.array([surgery, surgery, perioperative])
    indicator = discordance_indicator(predicted, reference)
    assert indicator.tolist() == [0, 1, 1]
    assert boundary_discordance(predicted, reference) == pytest.approx(2 / 3)


def test_discordance_summary_is_per_site() -> None:
    surgery = StageTriple(1, 0, 0).flat_index
    perioperative = StageTriple(2, 1, 0).flat_index
    predicted = np.array([surgery, perioperative, surgery, perioperative])
    reference = np.array([surgery, surgery, perioperative, perioperative])
    summary = discordance_summary(predicted, reference, ["A", "A", "B", "B"])
    assert summary.pooled == pytest.approx(50.0)
    assert summary.per_site == {"A": 50.0, "B": 50.0}
    assert summary.per_site_counts == {"A": 2, "B": 2}


def test_reclassification_improvement_components() -> None:
    reference = np.array([0, 1, 0, 1])
    candidate = np.array([1, 1, 0, 1])
    positive = np.array([1, 1, 0, 1])
    continuous = net_reclassification_improvement(reference, candidate, positive)
    assert continuous == pytest.approx(1 / 3)
    components = binary_nri(reference, candidate, positive)
    assert components["events"] == pytest.approx(1 / 3)
    assert components["non_events"] == pytest.approx(0.0)


def test_feasibility_metrics() -> None:
    from stagefm.data.staging import AchievableSet

    achievable = AchievableSet()
    infeasible = StageTriple(1, 3, 0).flat_index
    feasible = StageTriple(4, 3, 1).flat_index
    columns = np.array([feasible, feasible, infeasible, infeasible])
    assert feasibility_rate(columns, achievable) == pytest.approx(0.5)
    breakdown = feasibility_breakdown(columns, achievable, ["A", "A", "B", "B"])
    assert breakdown["pooled"] == pytest.approx(50.0)
    assert breakdown["per_site"] == {"A": 100.0, "B": 0.0}
    assert infeasible_examples(columns, achievable, limit=1) == ["T1N3M0"]


def test_staging_heads_loss_runs_on_every_axis() -> None:
    heads = StagingHeads(input_dim=12, config=OrdinalConfig(), ordinal=True)
    features = torch.randn(6, 12)
    outputs = heads(features)
    targets = {"T": torch.randint(0, 4, (6,)), "N": torch.randint(0, 4, (6,)), "M": torch.randint(0, 2, (6,))}
    loss = ordinal_loss(outputs, targets, ordinal=True)
    assert loss.per_axis.keys() == {"T", "N", "M"}
    assert float(loss.total.detach()) > 0
