"""Statistical procedures: DeLong intervals, bootstrap, McNemar, variance components."""

from __future__ import annotations

from .bootstrap import BootstrapResult, bootstrap_statistic, bootstrap_variance_components, paired_bootstrap, percentile_interval
from .correlation import pearson, scan_handling_test, spearman, spearman_with_bootstrap
from .delong import DeLongResult, auc_difference_test, auc_variance, delong_auc_ci, macro_auc_ci, midrank, placement_values
from .icc import VarianceComponents, anova_table, decompose, random_effects_decomposition, variance_reduction
from .mcnemar import McNemarResult, mcnemar, paired_difference_ci
from .multiplicity import BHResult, InteractionDecomposition, benjamini_hochberg, interaction_ratio, standardised_difference

__all__ = [
    "BHResult",
    "BootstrapResult",
    "DeLongResult",
    "InteractionDecomposition",
    "McNemarResult",
    "VarianceComponents",
    "anova_table",
    "auc_difference_test",
    "auc_variance",
    "benjamini_hochberg",
    "bootstrap_statistic",
    "bootstrap_variance_components",
    "decompose",
    "delong_auc_ci",
    "interaction_ratio",
    "macro_auc_ci",
    "mcnemar",
    "midrank",
    "paired_bootstrap",
    "paired_difference_ci",
    "pearson",
    "percentile_interval",
    "placement_values",
    "random_effects_decomposition",
    "scan_handling_test",
    "spearman",
    "spearman_with_bootstrap",
    "standardised_difference",
    "variance_reduction",
]
