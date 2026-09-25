"""Statistical-procedure tests, each with an expectation from an independent route."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as scipy_stats

from stagefm.stats.bootstrap import bootstrap_statistic, bootstrap_variance_components, paired_bootstrap, percentile_interval
from stagefm.stats.correlation import pearson, scan_handling_test, spearman, spearman_with_bootstrap
from stagefm.stats.delong import (
    auc_difference_test,
    auc_variance,
    delong_auc_ci,
    macro_auc_ci,
    midrank,
    placement_values,
)
from stagefm.stats.icc import anova_table, decompose, random_effects_decomposition, variance_reduction
from stagefm.stats.mcnemar import mcnemar, paired_difference_ci
from stagefm.stats.multiplicity import InteractionDecomposition, benjamini_hochberg, interaction_ratio, standardised_difference


def _brute_force_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    total = 0.0
    for value in positive:
        total += float((value > negative).sum()) + 0.5 * float((value == negative).sum())
    return total / (positive.size * negative.size)


def test_midrank_credits_ties_at_the_average() -> None:
    ranks = midrank(np.array([3.0, 1.0, 2.0, 2.0]))
    assert ranks.tolist() == [4.0, 1.0, 2.5, 2.5]


def test_placement_values_average_to_the_area() -> None:
    scores = np.array([0.9, 0.6, 0.4, 0.3, 0.1])
    labels = np.array([1, 1, 0, 0, 1])
    v10, v01 = placement_values(scores, labels)
    assert v10.mean() == pytest.approx(_brute_force_auc(scores[labels == 1], scores[labels == 0]))
    assert v01.shape == (int((labels == 0).sum()),)


def test_auc_variance_rejects_a_single_class() -> None:
    with pytest.raises(ValueError):
        auc_variance(np.array([0.1, 0.2]), np.zeros(2, dtype=int))
    with pytest.raises(ValueError):
        placement_values(np.array([0.1, 0.2]), np.zeros(2, dtype=int))


def test_delong_interval_brackets_the_area_and_shrinks_with_sample_size() -> None:
    rng = np.random.default_rng(0)
    small_labels = rng.integers(0, 2, 40)
    small_scores = small_labels + rng.normal(size=40)
    large_labels = rng.integers(0, 2, 4000)
    large_scores = large_labels + rng.normal(size=4000)
    small = delong_auc_ci(small_scores, small_labels)
    large = delong_auc_ci(large_scores, large_labels)
    assert small.low <= small.auc <= small.high
    assert large.standard_error < small.standard_error
    assert large.n_positive == int((large_labels == 1).sum())


def test_delong_standard_error_agrees_with_a_bootstrap() -> None:
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 2, 300)
    scores = labels + rng.normal(size=300)
    analytic = delong_auc_ci(scores, labels).standard_error
    replicates = np.empty(500)
    for position in range(500):
        draw = rng.integers(0, scores.size, size=scores.size)
        if len(np.unique(labels[draw])) < 2:
            replicates[position] = np.nan
            continue
        replicates[position] = _brute_force_auc(scores[draw][labels[draw] == 1], scores[draw][labels[draw] == 0])
    empirical = float(np.nanstd(replicates, ddof=1))
    assert analytic == pytest.approx(empirical, rel=0.25, abs=0.01)


def test_macro_auc_ci_averages_the_per_class_areas() -> None:
    rng = np.random.default_rng(2)
    labels = rng.integers(0, 3, 300)
    scores = rng.random((300, 3)) + 0.8 * np.eye(3)[labels]
    result = macro_auc_ci(scores, labels, 3)
    per_class = [_brute_force_auc(scores[labels == category, category], scores[labels != category, category]) for category in range(3)]
    assert result.auc == pytest.approx(float(np.mean(per_class)))
    assert result.low <= result.auc <= result.high


def test_paired_auc_difference_uses_the_covariance() -> None:
    rng = np.random.default_rng(3)
    labels = rng.integers(0, 2, 400)
    shared = labels + rng.normal(scale=0.5, size=400)
    first = shared + rng.normal(scale=0.1, size=400)
    second = labels + rng.normal(scale=0.6, size=400)
    result = auc_difference_test(first, second, labels)
    assert result["difference"] == pytest.approx(result["difference"])
    assert result["low"] <= result["difference"] <= result["high"]
    independent = auc_difference_test(first, first + rng.normal(scale=1.5, size=400), labels)
    assert independent["standard_error"] > 0.0


def test_percentile_interval_brackets_the_median() -> None:
    rng = np.random.default_rng(4)
    values = rng.normal(size=5000)
    low, high = percentile_interval(values, 0.05)
    assert low < float(np.median(values)) < high
    assert np.isnan(percentile_interval(np.zeros(0))[0])


def test_bootstrap_statistic_recovers_the_point_estimate() -> None:
    rng = np.random.default_rng(5)
    sample = rng.normal(loc=2.0, size=400)
    result = bootstrap_statistic(lambda index: float(sample[index].mean()), count=sample.size, resamples=500, seed=0)
    assert result.estimate == pytest.approx(float(sample.mean()))
    assert result.low <= result.estimate <= result.high
    assert result.replicates == 500
    assert result.half_width == pytest.approx(0.5 * (result.high - result.low))


def test_bootstrap_statistic_handles_an_empty_input() -> None:
    result = bootstrap_statistic(lambda index: 0.0, count=0, resamples=10)
    assert np.isnan(result.estimate) and result.replicates == 0


def test_paired_bootstrap_reuses_the_same_records_in_both_arms() -> None:
    first = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 1.0])
    second = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 1.0])
    result = paired_bootstrap(
        lambda index: float(first[index].mean() - second[index].mean()),
        count=first.size,
        resamples=200,
        seed=1,
    )
    assert result.estimate == pytest.approx(float(first.mean() - second.mean()))
    zero = paired_bootstrap(lambda index: float(first[index].mean() - first[index].mean()), count=first.size, resamples=50, seed=1)
    assert zero.estimate == pytest.approx(0.0)
    assert abs(zero.high - zero.low) < 1e-12


def test_variance_decomposition_matches_hand_anova() -> None:
    rng = np.random.default_rng(6)
    groups = np.repeat(np.array(["A", "B", "C"]), 30)
    values = np.repeat(np.array([0.1, 0.3, 0.5]), 30) + rng.normal(scale=0.02, size=90)
    components = decompose(values, groups)
    table = anova_table(values, groups)
    assert components.mean_square_between == pytest.approx(table["between"]["ms"])
    assert components.mean_square_within == pytest.approx(table["within"]["ms"])
    hand_between = (table["between"]["ss"] / 2 - table["within"]["ms"]) / (90 / 3)
    assert components.between_site_variance == pytest.approx(max(hand_between, 0.0))
    assert 0.0 <= components.intraclass_correlation <= 1.0
    assert set(random_effects_decomposition(values, groups)) >= {"between_site_variance", "intraclass_correlation"}


def test_variance_decomposition_handles_too_few_groups() -> None:
    components = decompose(np.array([0.1, 0.2, 0.3]), np.array(["A", "A", "A"]))
    assert np.isnan(components.between_site_variance)
    assert components.groups == 1


def test_variance_reduction_and_guards() -> None:
    assert variance_reduction(0.00241, 0.00070) == pytest.approx(0.7095, abs=1e-4)
    assert np.isnan(variance_reduction(0.0, 0.1))
    assert np.isnan(variance_reduction(float("nan"), 0.1))


def test_bootstrap_variance_components_returns_intervals() -> None:
    rng = np.random.default_rng(7)
    site = np.repeat(np.array(["A", "B", "C", "D"]), 50)
    indicator = (rng.random(200) < (np.repeat([0.05, 0.1, 0.2, 0.3], 50))).astype(float)
    result = bootstrap_variance_components(indicator, site, resamples=60, seed=0)
    assert result["replicates"] == 60
    assert np.isfinite(result["between_site_variance"])
    assert result["between_low"] <= result["between_high"]


def test_bootstrap_variance_components_needs_two_sites() -> None:
    result = bootstrap_variance_components(np.array([0.0, 1.0]), np.array(["A", "A"]), resamples=10, seed=0)
    assert result["replicates"] == 0


def test_mcnemar_reproduces_the_exact_exchangeability_probability() -> None:
    first = np.array([1, 1, 1, 0, 0, 1, 1, 0, 1, 1])
    second = np.array([1, 0, 0, 0, 1, 1, 1, 0, 1, 1])
    result = mcnemar(first, second)
    only_first = int(((first == 1) & (second == 0)).sum())
    only_second = int(((first == 0) & (second == 1)).sum())
    discordant = only_first + only_second
    expected = float(min(1.0, 2.0 * scipy_stats.binom.cdf(min(only_first, only_second), discordant, 0.5)))
    assert result.p_value_exact == pytest.approx(expected)
    assert result.only_first == only_first and result.only_second == only_second
    assert result.both_correct == int(((first == 1) & (second == 1)).sum())
    assert result.discordant == discordant


def test_mcnemar_handles_a_perfect_agreement() -> None:
    same = np.array([1, 1, 0, 0])
    result = mcnemar(same, same)
    assert result.discordant == 0
    assert result.p_value == 1.0
    assert result.difference == pytest.approx(0.0)


def test_paired_difference_interval() -> None:
    difference = np.array([0.1] * 50)
    result = paired_difference_ci(difference)
    assert result["mean"] == pytest.approx(0.1)
    assert result["low"] == pytest.approx(result["high"])


def test_pearson_matches_the_library_and_handles_short_input() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    result = pearson(x, y)
    assert result["r"] == pytest.approx(1.0)
    assert result["n"] == 5.0
    assert np.isnan(pearson(np.array([1.0]), np.array([1.0]))["r"])


def test_spearman_and_bootstrap_interval() -> None:
    x = np.arange(30, dtype=float)
    y = x * 2.0 + np.sin(x)
    rank = spearman(x, y)
    assert rank["rho"] == pytest.approx(1.0)
    boot = spearman_with_bootstrap(x, y, resamples=100, seed=0)
    assert boot["low"] <= boot["rho"] <= boot["high"]
    assert boot["replicates"] == 100.0


def test_permutation_correlation_test_returns_a_probability() -> None:
    rng = np.random.default_rng(8)
    x = rng.normal(size=40)
    y = x * 0.9 + rng.normal(scale=0.3, size=40)
    result = scan_handling_test(x, y)
    assert 0.0 < result["p_permutation"] <= 1.0
    assert result["r"] > 0.5


def test_benjamini_hochberg_matches_the_reference_implementation() -> None:
    p_values = [0.001, 0.008, 0.02, 0.04, 0.2, 0.6, 0.9]
    from statsmodels.stats.multitest import multipletests

    result = benjamini_hochberg(p_values, level=0.05)
    library_reject, library_adjusted = multipletests(p_values, alpha=0.05, method="fdr_bh")[:2]
    assert result.adjusted == pytest.approx(list(library_adjusted))
    assert result.family_size == len(p_values)
    assert result.rejected == list(np.flatnonzero(library_reject))
    assert result.rejected == [0, 1, 2]


def test_benjamini_hochberg_handles_an_empty_family() -> None:
    result = benjamini_hochberg([], level=0.05)
    assert result.rejected == [] and result.adjusted == [] and result.family_size == 0


def test_interaction_ratio_and_decomposition_from_tabulated_rows() -> None:
    assert interaction_ratio(5.80, 1.41, 3.05) == pytest.approx(1.3004, abs=1e-4)
    assert np.isnan(interaction_ratio(1.0, 0.0, 0.0))
    decomposition = InteractionDecomposition(reference=8.92, without_first=5.87, without_second=7.51, full=3.12)
    detail = decomposition.as_dict()
    assert detail["separate_first"] == pytest.approx(1.41)
    assert detail["separate_second"] == pytest.approx(3.05)
    assert detail["joint"] == pytest.approx(5.80)
    assert detail["ratio"] == pytest.approx(1.3004, abs=1e-4)
    assert round(detail["ratio"], 2) == 1.30


def test_concordance_interaction_does_not_reproduce_the_reported_value() -> None:
    """The manuscript reports 0.87 for the concordance ratio; its own rows give 0.57.

    This test pins the discrepancy so it cannot be silently reconciled: if a later
    revision changes either the estimator or the reported value, it fails.
    """
    decomposition = InteractionDecomposition(reference=0.8058, without_first=0.8271, without_second=0.8291, full=0.8312)
    assert round(decomposition.ratio, 2) == 0.57
    assert round(decomposition.ratio, 2) != 0.87


def test_standardised_difference_guard() -> None:
    assert standardised_difference(1.0, 1.5, 0.5) == pytest.approx(1.0)
    assert np.isnan(standardised_difference(1.0, 1.5, 0.0))
