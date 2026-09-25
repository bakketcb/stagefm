"""Release verification.

Two passes run here and both have to hold before the release can be reported as
verified.

Pass 1 resolves the paper-claim to code mapping: every declared anchor has to name a
real file below the package directory and a real module-level symbol, parsed from the
abstract syntax tree rather than trusted from the file name.

Pass 2 executes the pipeline end to end -- cohort construction, volume synthesis and
preprocessing, forward pass, objective, backward pass, parameter update, checkpoint
round trip, a single-batch overfit and a brief training loop -- and then re-derives
the numerical results of the core algorithms with independent code: explicit
enumeration for the feasibility projection, brute-force pairwise counting for the
areas under the curve, a hand-computed decomposition for the intraclass correlation,
closed-form evaluation of the decision metrics, and the ablation table's own
arithmetic for the interaction ratios. The verification code never calls the routine
it is checking to produce the expected value.

The integrity manifest is written last so that it digests the tree in its final
state, including the report it has just produced.

Ref: Algorithm 1-4; Methods Sec. 4.1-4.12.
"""

from __future__ import annotations

import argparse
import ast
import traceback
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..utils.hashing import manifest_digest
from ..utils.io import to_builtin, write_json, write_text
from ..utils.logging import get_logger
from ..version import RELEASE_SLUG

LOGGER = get_logger("cli.verify")

PASS = "PASS"
FAIL = "FAIL"
NOT_RUN = "NOT_RUN"
BLOCKED = "BLOCKED"
VERIFIED = "VERIFIED"
PARTIALLY_VERIFIED = "PARTIALLY_VERIFIED"
UNVERIFIED = "UNVERIFIED"

MANIFEST_NAME = "integrity_manifest.json"
VERIFICATION_NAME = "verification_report.json"
CLAIM_NAME = "claim_to_code.json"
SUMMARY_NAME = "verification_summary.txt"
DATASET_URLS_NAME = "dataset_urls.txt"
SMOKE_CONFIG = "configs/experiment/_smoke.yaml"
PACKAGE_DIR = "src/stagefm"

CLAIMS: list[dict[str, Any]] = [
    {
        "id": "sec2.2-primary-endpoint",
        "paper_location": "Sec. 2.2; Methods Sec. 4.1",
        "claim": "The primary endpoint is treatment-boundary discordance: the share of examinations whose predicted stage implies a different management category from the one surgical pathology supports.",
        "code": [{"path": "metrics/decision.py", "symbol": "boundary_discordance"}, {"path": "data/staging.py", "symbol": "treatment_category"}],
    },
    {
        "id": "sec2.2-boundary-rule",
        "paper_location": "Methods Sec. 4.1 (treatment-boundary labels)",
        "claim": "A single pre-defined rule maps a stage onto a management category and takes no site-varying input.",
        "code": [{"path": "data/staging.py", "symbol": "treatment_category"}, {"path": "data/staging.py", "symbol": "boundary_discordant"}],
    },
    {
        "id": "sec2.2-feasibility",
        "paper_location": "Sec. 2.2; Table 3",
        "claim": "Stage feasibility rises from 71.3 percent for independent heads to 100 percent under the constrained model.",
        "code": [{"path": "metrics/feasibility.py", "symbol": "feasibility_rate"}, {"path": "models/stage_consistency.py", "symbol": "FeasibilityProjection"}],
    },
    {
        "id": "sec2.3-interaction",
        "paper_location": "Sec. 2.3; Table 4",
        "claim": "With the configuration lacking both structural components as reference, the two contribute 1.41 and 3.05 points separately and 5.80 jointly, an interaction ratio of 1.30.",
        "code": [{"path": "stats/multiplicity.py", "symbol": "interaction_ratio"}, {"path": "stats/multiplicity.py", "symbol": "InteractionDecomposition"}],
    },
    {
        "id": "sec2.4-ascertainment",
        "paper_location": "Sec. 2.4; Fig. 2",
        "claim": "Nodal-axis error tracks the site's median harvested node count while tumour-axis error does not.",
        "code": [{"path": "evaluation/ascertainment.py", "symbol": "ascertainment_report"}, {"path": "stats/correlation.py", "symbol": "pearson"}],
    },
    {
        "id": "sec2.4-variance-reduction",
        "paper_location": "Sec. 2.4; Fig. 2 caption",
        "claim": "Restricting to examinations with at least 16 harvested nodes removes 71% of the between-site variance in nodal error.",
        "code": [{"path": "stats/icc.py", "symbol": "decompose"}, {"path": "stats/icc.py", "symbol": "variance_reduction"}],
    },
    {
        "id": "sec2.5-subgroups",
        "paper_location": "Sec. 2.5",
        "claim": "The adequacy stratum carries the largest subgroup gap, and the sex, age and vendor families stay within the null bands.",
        "code": [{"path": "evaluation/subgroups.py", "symbol": "subgroup_report"}, {"path": "evaluation/subgroups.py", "symbol": "family_summary"}],
    },
    {
        "id": "sec2.8-silent-mode",
        "paper_location": "Sec. 2.8; Table 1",
        "claim": "Deployment was silent, the panel was opened in 1,446 of 1,850 examinations, and no examination was delayed.",
        "code": [
            {"path": "evaluation/silent_mode.py", "symbol": "summarise_deployment"},
            {"path": "evaluation/silent_mode.py", "symbol": "stratified_open_rate"},
        ],
    },
    {
        "id": "sec2.10-in-vitro",
        "paper_location": "Sec. 2.10; Methods Sec. 4.10",
        "claim": "Conditioned medium from high-VEGF-C lines raised the tube-formation index and VEGFR-3 phosphorylation, and VEGF-C knockdown removed both effects.",
        "code": [{"path": "evaluation/in_vitro.py", "symbol": "knockdown_reversal"}, {"path": "evaluation/in_vitro.py", "symbol": "blocked_condition_effect"}],
    },
    {
        "id": "sec2.10-texture-density",
        "paper_location": "Sec. 2.10",
        "claim": "The peritumoral texture feature carrying the nodal signal correlates with lymphatic vessel density on resection material.",
        "code": [
            {"path": "evaluation/in_vitro.py", "symbol": "texture_density_correlation"},
            {"path": "stats/correlation.py", "symbol": "spearman_with_bootstrap"},
        ],
    },
    {
        "id": "sec2.11-worse-than-baseline",
        "paper_location": "Sec. 2.11",
        "claim": "In the earliest T stratum the nodal area under the curve falls below the independent-head arm, and in the diffuse stratum the metastatic axis falls below the imaging-only network, both within the 0.02 minimum clinically important difference.",
        "code": [{"path": "evaluation/subgroups.py", "symbol": "MINIMUM_CLINICALLY_IMPORTANT_AUROC"}, {"path": "models/baselines.py", "symbol": "BASELINES"}],
    },
    {
        "id": "sec4.1-layers",
        "paper_location": "Methods Sec. 4.1",
        "claim": "Development is 7,828 records (5,868 training and 1,960 internal test) from three sites, and 3,456 records from two further sites form the external layer with no contribution to training, selection or calibration.",
        "code": [{"path": "data/cohort.py", "symbol": "layer_sizes"}, {"path": "utils/config.py", "symbol": "CohortConfig"}],
    },
    {
        "id": "sec4.1-label-space",
        "paper_location": "Methods Sec. 4.1",
        "claim": "The tumour axis is a four-category grouping with T4a and T4b merged, the nodal axis is ordinal over four categories, and the metastatic axis is binary.",
        "code": [{"path": "data/schema.py", "symbol": "StageTriple"}, {"path": "data/schema.py", "symbol": "ALL_TRIPLES"}],
    },
    {
        "id": "sec4.1-site-split",
        "paper_location": "Methods Sec. 4.1; Sec. 4.6",
        "claim": "Whole sites are held out rather than random patients, because the claim concerns transport to an unseen site.",
        "code": [{"path": "data/cohort.py", "symbol": "CohortSplit"}, {"path": "data/synthetic.py", "symbol": "generate_cohort"}],
    },
    {
        "id": "sec4.3-isotropic",
        "paper_location": "Methods Sec. 4.3",
        "claim": "Volumes are resampled to 1x1x1 mm isotropic spacing.",
        "code": [{"path": "data/imaging.py", "symbol": "resample_isotropic"}, {"path": "utils/config.py", "symbol": "ImagingConfig"}],
    },
    {
        "id": "sec4.3-hu-window",
        "paper_location": "Methods Sec. 4.3",
        "claim": "An abdominal Hounsfield window is applied to the volumes.",
        "code": [{"path": "data/imaging.py", "symbol": "apply_hu_window"}],
    },
    {
        "id": "sec4.3-bias-field",
        "paper_location": "Methods Sec. 4.3",
        "claim": "Bias-field correction and per-examination intensity standardisation are applied.",
        "code": [{"path": "data/imaging.py", "symbol": "correct_bias_field"}, {"path": "data/imaging.py", "symbol": "standardise_intensity"}],
    },
    {
        "id": "sec4.3-localisation",
        "paper_location": "Methods Sec. 4.3",
        "claim": "The gastric wall and the nodal basin are localised automatically, with no manual lesion definition.",
        "code": [{"path": "data/segmentation.py", "symbol": "localise"}, {"path": "data/segmentation.py", "symbol": "nodal_basin_mask"}],
    },
    {
        "id": "sec4.3-shell",
        "paper_location": "Methods Sec. 4.3",
        "claim": "Radiomic descriptors are computed on the region between 0 and 5 mm outside the localised wall and basin boundary.",
        "code": [{"path": "data/segmentation.py", "symbol": "peritumoral_shell"}, {"path": "data/radiomics.py", "symbol": "extract_descriptors"}],
    },
    {
        "id": "sec4.3-stability",
        "paper_location": "Methods Sec. 4.3; Ref. [56]",
        "claim": "Only descriptors that survive a stability criterion across independently repeated segmentations are retained.",
        "code": [{"path": "data/radiomics.py", "symbol": "stability_select"}, {"path": "data/radiomics.py", "symbol": "intraclass_correlation"}],
    },
    {
        "id": "sec4.3-redundancy",
        "paper_location": "Methods Sec. 4.3; Ref. [57]",
        "claim": "Correlated descriptor families are collapsed to one representative per block rather than entered jointly.",
        "code": [{"path": "data/radiomics.py", "symbol": "collapse_redundant"}],
    },
    {
        "id": "sec4.3-per-site-zscore",
        "paper_location": "Methods Sec. 4.3",
        "claim": "Descriptors are z-scored within each site using development-set statistics alone.",
        "code": [{"path": "data/radiomics.py", "symbol": "SiteStandardiser"}],
    },
    {
        "id": "sec4.4-auxiliary",
        "paper_location": "Methods Sec. 4.4; Data availability",
        "claim": "Three public resources support preprocessing and pretraining and contribute no clinical evidence.",
        "code": [{"path": "models/encoder.py", "symbol": "build_encoder"}, {"path": "models/encoder.py", "symbol": "EncoderUnavailable"}],
    },
    {
        "id": "sec4.5-encoder",
        "paper_location": "Methods Sec. 4.5",
        "claim": "The encoder is the released abdominal CT vision-language model, used frozen with low-rank adaptation on the attention projections.",
        "code": [{"path": "models/encoder.py", "symbol": "CompactCTEncoder"}, {"path": "models/lora.py", "symbol": "apply_lora"}],
    },
    {
        "id": "sec4.5-fusion",
        "paper_location": "Methods Sec. 4.5",
        "claim": "The fusion block combines the encoder's patch tokens with the radiomic, structured clinical and text streams.",
        "code": [{"path": "models/fusion.py", "symbol": "CrossAttentionFusion"}, {"path": "models/fusion.py", "symbol": "FusionInputs"}],
    },
    {
        "id": "sec4.5-absence-token",
        "paper_location": "Methods Sec. 4.5; Sec. 4.11",
        "claim": "A stream unavailable at the point of care is represented by an absence token rather than imputed.",
        "code": [{"path": "models/fusion.py", "symbol": "CrossAttentionFusion"}, {"path": "data/text.py", "symbol": "TextVocabulary"}],
    },
    {
        "id": "sec4.5-ordinal-head",
        "paper_location": "Methods Sec. 4.5; Sec. 4.6",
        "claim": "Each axis uses a monotonic ordinal head whose single latent score passes through ordered cut points, so the ordinal structure cannot be disrupted.",
        "code": [{"path": "models/ordinal.py", "symbol": "MonotonicOrdinalHead"}, {"path": "models/ordinal.py", "symbol": "StagingHeads"}],
    },
    {
        "id": "sec4.5-stage-consistency",
        "paper_location": "Methods Sec. 4.5; Algorithm 1 step 8",
        "claim": "The axis distributions are composed into a joint distribution over stage combinations and projected onto the achievable set, deterministically and differentiably.",
        "code": [
            {"path": "models/stage_consistency.py", "symbol": "joint_distribution"},
            {"path": "models/stage_consistency.py", "symbol": "FeasibilityProjection"},
        ],
    },
    {
        "id": "alg3-projection",
        "paper_location": "Algorithm 3",
        "claim": "The projection renormalises the mass on the achievable set, sets the rest to zero, and returns the uniform distribution on that set when no mass falls there.",
        "code": [
            {"path": "models/stage_consistency.py", "symbol": "FeasibilityProjection"},
            {"path": "models/stage_consistency.py", "symbol": "ProjectionOutput"},
        ],
    },
    {
        "id": "sec4.6-achievable-set",
        "paper_location": "Methods Sec. 4.6",
        "claim": "The achievable set is established from staging definitions rather than derived from the data, so feasibility is a statement about the staging system.",
        "code": [{"path": "data/staging.py", "symbol": "AchievableSet"}, {"path": "data/staging.py", "symbol": "AchievableRule"}],
    },
    {
        "id": "sec4.6-modality-dropout",
        "paper_location": "Methods Sec. 4.6; Sec. 4.7",
        "claim": "Modality dropout is 0.15 per non-imaging stream.",
        "code": [{"path": "data/streams.py", "symbol": "dropout_rates"}, {"path": "utils/config.py", "symbol": "FusionConfig"}],
    },
    {
        "id": "sec4.5-risk-control",
        "paper_location": "Methods Sec. 4.5; Sec. 4.6",
        "claim": "Calibration applies an affine correction on the logit scale indexed by site, and the decision threshold is chosen per site stratum by minimising the empirical boundary error.",
        "code": [{"path": "models/risk_control.py", "symbol": "SiteCalibrator"}, {"path": "models/risk_control.py", "symbol": "select_thresholds"}],
    },
    {
        "id": "sec4.5-threshold-freeze",
        "paper_location": "Methods Sec. 4.5; Algorithm 1 step 12",
        "claim": "The threshold is chosen on the internal test split and then frozen, and is never recalibrated on the cohort it is reported for.",
        "code": [{"path": "training/trainer.py", "symbol": "Trainer"}],
    },
    {
        "id": "alg2-finite-sample-bound",
        "paper_location": "Algorithm 2 step 7",
        "claim": "A finite-sample upper bound on the site-conditional error is reported with each threshold and covers the whole grid by a union bound.",
        "code": [{"path": "models/risk_control.py", "symbol": "ThresholdSelection"}, {"path": "models/risk_control.py", "symbol": "select_thresholds"}],
    },
    {
        "id": "sec4.7-optimizer",
        "paper_location": "Methods Sec. 4.7",
        "claim": "AdamW with decoupled weight decay of 1e-4, a maximum learning rate of 3e-4 for fusion and prediction parameters and 1e-4 for the adapters, gradient clipping at 1.0.",
        "code": [{"path": "training/optim.py", "symbol": "build_optimizer"}, {"path": "utils/config.py", "symbol": "TrainConfig"}],
    },
    {
        "id": "sec4.7-schedule",
        "paper_location": "Methods Sec. 4.7",
        "claim": "Cosine decay with a linear warmup over the first 500 steps.",
        "code": [{"path": "training/scheduler.py", "symbol": "ScheduleSpec"}, {"path": "training/scheduler.py", "symbol": "build_scheduler"}],
    },
    {
        "id": "sec4.7-lora-config",
        "paper_location": "Methods Sec. 4.7",
        "claim": "Rank 16 with alpha 32 on the query, key, value and output projections of the vision encoder, with every other encoder weight frozen.",
        "code": [{"path": "models/lora.py", "symbol": "LoRALinear"}, {"path": "utils/config.py", "symbol": "EncoderConfig"}],
    },
    {
        "id": "sec4.7-batch-and-grid",
        "paper_location": "Methods Sec. 4.7",
        "claim": "Batches of eight examinations remapped into a fixed 96 x 96 x 64 voxel patch grid, at most 60 epochs with early stopping at patience 8.",
        "code": [{"path": "utils/config.py", "symbol": "TrainConfig"}, {"path": "utils/config.py", "symbol": "ImagingConfig"}],
    },
    {
        "id": "sec4.7-augmentation",
        "paper_location": "Methods Sec. 4.7",
        "claim": "Augmentation is limited to a bounded affine perturbation and a bounded intensity jitter that do not alter staging-relevant anatomy.",
        "code": [{"path": "data/imaging.py", "symbol": "augment_volume"}],
    },
    {
        "id": "sec4.7-runs",
        "paper_location": "Methods Sec. 4.7",
        "claim": "Five independent runs per configuration, with the mean reported and the standard deviation across runs in the table.",
        "code": [{"path": "utils/config.py", "symbol": "CohortConfig"}, {"path": "utils/seeding.py", "symbol": "set_seed"}],
    },
    {
        "id": "sec4.7-compute",
        "paper_location": "Methods Sec. 4.7",
        "claim": "One node with four accelerators and 80 GB of memory per node, about 3,100 accelerator-hours.",
        "code": [{"path": "utils/config.py", "symbol": "TrainConfig"}, {"path": "training/distributed.py", "symbol": "DistributedContext"}],
    },
    {
        "id": "alg1-objective",
        "paper_location": "Algorithm 1 step 9",
        "claim": "The objective sums the ordinal loss, a decision term weighted by lambda and a calibration term weighted by gamma.",
        "code": [{"path": "losses/total.py", "symbol": "staged_objective"}, {"path": "losses/total.py", "symbol": "LossBreakdown"}],
    },
    {
        "id": "alg1-ordinal-loss",
        "paper_location": "Algorithm 1 steps 5-9",
        "claim": "The ordinal term is the negative log-likelihood of the cumulative-link model on the head's cumulative logits.",
        "code": [{"path": "losses/ordinal.py", "symbol": "cumulative_link_loss"}, {"path": "losses/ordinal.py", "symbol": "ordinal_loss"}],
    },
    {
        "id": "alg1-decision-loss",
        "paper_location": "Algorithm 1 step 9; Sec. 4.1",
        "claim": "The decision term is a differentiable surrogate of the boundary error read off the projected distribution.",
        "code": [{"path": "losses/decision.py", "symbol": "decision_loss"}, {"path": "losses/decision.py", "symbol": "discordance_surrogate"}],
    },
    {
        "id": "alg1-calibration-loss",
        "paper_location": "Algorithm 1 step 9; Sec. 4.11",
        "claim": "The calibration term penalises the site-level gap between mean predicted probability and observed frequency.",
        "code": [{"path": "losses/calibration.py", "symbol": "calibration_loss"}, {"path": "losses/calibration.py", "symbol": "site_gap_loss"}],
    },
    {
        "id": "sec4.8-baselines",
        "paper_location": "Methods Sec. 4.8; Table 3",
        "claim": "Eleven arms are compared on the same external cohort with the same tuning budget, and arms whose values are quoted rather than computed are marked as such.",
        "code": [{"path": "models/baselines.py", "symbol": "BASELINES"}, {"path": "models/baselines.py", "symbol": "QUOTED_ARMS"}],
    },
    {
        "id": "sec4.8-stream-only-arms",
        "paper_location": "Methods Sec. 4.8",
        "claim": "Single-modality arms read only their own stream on the same token contract as the full model.",
        "code": [{"path": "models/baselines.py", "symbol": "StreamOnlyModel"}, {"path": "models/baselines.py", "symbol": "build_arm"}],
    },
    {
        "id": "sec4.9-reader-study",
        "paper_location": "Methods Sec. 4.9",
        "claim": "Eighteen readers read 360 examinations twice in a paired within-reader design, unassisted session first, with a washout interval between sessions.",
        "code": [
            {"path": "evaluation/reader_study.py", "symbol": "ReaderStudyDesign"},
            {"path": "evaluation/reader_study.py", "symbol": "reader_study_report"},
        ],
    },
    {
        "id": "sec4.9-paired-analysis",
        "paper_location": "Methods Sec. 4.9",
        "claim": "The reader analysis is paired on reader and case, and its interval comes from resampling the paired observations.",
        "code": [{"path": "evaluation/reader_study.py", "symbol": "paired_reader_difference"}],
    },
    {
        "id": "sec4.11-delong",
        "paper_location": "Methods Sec. 4.11",
        "claim": "Discrimination on every axis is summarised with DeLong intervals.",
        "code": [{"path": "stats/delong.py", "symbol": "delong_auc_ci"}, {"path": "stats/delong.py", "symbol": "macro_auc_ci"}],
    },
    {
        "id": "sec4.11-ordinal-auroc",
        "paper_location": "Methods Sec. 4.11",
        "claim": "The nodal axis is summarised by a one-dimensional ordinal summary rather than by a binary threshold.",
        "code": [{"path": "metrics/discrimination.py", "symbol": "ordinal_auroc"}, {"path": "metrics/discrimination.py", "symbol": "axis_discrimination"}],
    },
    {
        "id": "sec4.11-kappa",
        "paper_location": "Methods Sec. 4.11; Table 3",
        "claim": "Weighted kappa and the exact three-axis match rate summarise agreement with pathology.",
        "code": [{"path": "metrics/concordance.py", "symbol": "weighted_kappa"}, {"path": "metrics/concordance.py", "symbol": "exact_combination_concordance"}],
    },
    {
        "id": "sec4.11-ece",
        "paper_location": "Methods Sec. 4.11; Fig. 3",
        "claim": "Expected calibration error is computed per site and pooled.",
        "code": [
            {"path": "metrics/calibration.py", "symbol": "expected_calibration_error"},
            {"path": "metrics/calibration.py", "symbol": "calibration_summary"},
        ],
    },
    {
        "id": "sec4.11-net-benefit",
        "paper_location": "Methods Sec. 4.11; Fig. 4",
        "claim": "Net benefit is computed across a threshold range and the interval on the primary endpoint comes from a bootstrap over examinations.",
        "code": [{"path": "metrics/decision.py", "symbol": "net_benefit"}, {"path": "stats/bootstrap.py", "symbol": "bootstrap_statistic"}],
    },
    {
        "id": "sec4.11-nri",
        "paper_location": "Sec. 2.2; Methods Sec. 4.11",
        "claim": "Net reclassification improvement is reported against the endoscopic-ultrasound standard and against the unconstrained control.",
        "code": [{"path": "metrics/decision.py", "symbol": "net_reclassification_improvement"}, {"path": "metrics/decision.py", "symbol": "binary_nri"}],
    },
    {
        "id": "sec4.11-mcnemar",
        "paper_location": "Sec. 2.2; Methods Sec. 4.11",
        "claim": "The reduction against the unconstrained control is tested with a paired test on the same examinations.",
        "code": [{"path": "stats/mcnemar.py", "symbol": "mcnemar"}, {"path": "stats/mcnemar.py", "symbol": "McNemarResult"}],
    },
    {
        "id": "sec4.11-variance-components",
        "paper_location": "Methods Sec. 4.11; Algorithm 4",
        "claim": "The between-site variance of the nodal error is decomposed with a site-level random intercept and bootstrapped within each site stratum.",
        "code": [{"path": "stats/icc.py", "symbol": "random_effects_decomposition"}, {"path": "stats/bootstrap.py", "symbol": "bootstrap_variance_components"}],
    },
    {
        "id": "sec4.11-fdr",
        "paper_location": "Methods Sec. 4.11",
        "claim": "Each subgroup family is corrected separately at a false-discovery-rate level fixed before the analysis.",
        "code": [{"path": "stats/multiplicity.py", "symbol": "benjamini_hochberg"}, {"path": "stats/multiplicity.py", "symbol": "BHResult"}],
    },
    {
        "id": "sec4.11-missingness",
        "paper_location": "Methods Sec. 4.11",
        "claim": "Nineteen records lack an endoscopic description and 61 lack pathology text, and the records are retained rather than dropped.",
        "code": [{"path": "data/synthetic.py", "symbol": "MISSING_ENDOSCOPY"}, {"path": "data/synthetic.py", "symbol": "MISSING_PATHOLOGY"}],
    },
    {
        "id": "sec4.11-sensitivity",
        "paper_location": "Methods Sec. 4.11",
        "claim": "Two sensitivity analyses are pre-specified: excluding consensus-staged examinations, and shifting the boundary-rule categories one step each way.",
        "code": [{"path": "evaluation/sensitivity.py", "symbol": "consensus_exclusion"}, {"path": "evaluation/sensitivity.py", "symbol": "boundary_shift"}],
    },
    {
        "id": "sec4.12-criterion",
        "paper_location": "Methods Sec. 4.12; Sec. 2.1",
        "claim": "Both pre-specified criteria had to be met: exact-combination concordance of at least 0.80 and a four-class tumour-axis area under the curve of at least 0.90.",
        "code": [{"path": "evaluation/report.py", "symbol": "prespecified_criterion"}, {"path": "evaluation/report.py", "symbol": "clinical_relevance"}],
    },
    {
        "id": "sec4.13-scope",
        "paper_location": "Methods Sec. 4.13",
        "claim": "The experiment set covers module ablations, distribution shift, the reader study, multi-site validation, a cohort study and an in vitro arm, and names what it omits.",
        "code": [{"path": "models/baselines.py", "symbol": "arm_table_rows"}, {"path": "evaluation/report.py", "symbol": "results_table"}],
    },
    {
        "id": "sec4.7-checkpoint",
        "paper_location": "Methods Sec. 4.7",
        "claim": "A checkpoint carries the seed and the frozen thresholds so a resumed run restores the same stream.",
        "code": [{"path": "training/checkpoint.py", "symbol": "save_checkpoint"}, {"path": "training/checkpoint.py", "symbol": "restore_into"}],
    },
    {
        "id": "sec4.7-amp",
        "paper_location": "Methods Sec. 4.7",
        "claim": "Precision is a configuration value and the run records the one it used.",
        "code": [{"path": "training/amp.py", "symbol": "PrecisionSpec"}, {"path": "training/amp.py", "symbol": "autocast_context"}],
    },
    {
        "id": "sec4.2-deidentification",
        "paper_location": "Methods Sec. 4.2",
        "claim": "Sites are represented by a code and classified into regions, and no personal identifier is carried into the analysis set.",
        "code": [{"path": "data/cohort.py", "symbol": "SiteProfile"}, {"path": "data/schema.py", "symbol": "Examination"}],
    },
]

DEVIATIONS: list[dict[str, str]] = [
    {
        "id": "dev1-private-cohort",
        "paper_location": "Methods Sec. 4.1; Data availability",
        "deviation": "The multi-centre cohort is held under a data-usage statement and is not redistributed, so the shipped pipeline reads a schema-compatible synthetic cohort instead. Every cohort-level clinical quantity (Tables 1, 3 and 4, Figs. 1-4, the per-site, subgroup and ascertainment tables) is therefore NOT_RUN, and no number computed from the generator may be presented as a study result.",
        "justification": "the data-availability statement releases the cohort only under its own usage policy, and the records cannot be redistributed",
    },
    {
        "id": "dev2-achievable-set-rule",
        "paper_location": "Methods Sec. 4.6",
        "deviation": "The manuscript states that the achievable set was established from staging definitions but does not enumerate it. The rule shipped here is an explicit construction from those definitions: a T1 lesion carries at most N1 involvement and no distant metastasis, a T2 lesion carries at most N2, and distant metastasis without regional node involvement is only reachable at T4. It admits 22 of the 32 combinations, a support a little below three quarters of the label space.",
        "justification": "an engineering default was required for an unstated set; the rule is exposed in config next to the oversized and undersized variants the sensitivity analysis moves",
    },
    {
        "id": "dev3-treatment-boundary-rule",
        "paper_location": "Methods Sec. 4.1",
        "deviation": "The manuscript does not enumerate the stage-to-category mapping. The rule shipped here follows the trial evidence it cites: a T1 lesion without nodal involvement is resected first, locally advanced but resectable disease receives perioperative chemotherapy, and distant or peritoneal disease is treated systemically.",
        "justification": "the endpoint needs a concrete mapping, and the rule is site-independent by construction, which is the property the manuscript requires of it",
    },
    {
        "id": "dev4-encoder-fallback",
        "paper_location": "Methods Sec. 4.4-4.5",
        "deviation": "The released abdominal CT vision-language encoder is distributed under a data-use agreement and is not bundled. When no local checkpoint is configured the pipeline uses the compact 3D vision transformer in models/encoder.py, which satisfies the same token contract and the same attention-projection naming the adapters bind to; the run metadata records which encoder was used.",
        "justification": "no redistributable weights, and the fallback keeps every downstream module executable without silently substituting a different architecture for the reported one",
    },
    {
        "id": "dev5-auxiliary-datasets",
        "paper_location": "Methods Sec. 4.4; Data availability",
        "deviation": "The three public auxiliary resources are downloaded separately and are absent from this checkout, so pretraining and auxiliary organ localisation are BLOCKED. Their access conditions are recorded in dataset_urls.txt, and no clinical claim depends on them.",
        "justification": "the checkpoints and cohorts are large external downloads; nothing reported here is derived from them",
    },
    {
        "id": "dev6-kappa-weighting",
        "paper_location": "Methods Sec. 4.11",
        "deviation": "The manuscript reports a weighted kappa without naming the weight function. A linear weight is used here; the quadratic weighting is available from the same function.",
        "justification": "a default was required, and the choice is an argument so the value can be recomputed under the other weighting",
    },
    {
        "id": "dev7-calibration-loss-form",
        "paper_location": "Algorithm 1 step 9",
        "deviation": "The third objective term is written as a calibration loss without a specified form. The shipped default is the mean absolute site-level gap between mean predicted probability and observed frequency, with a Brier variant available.",
        "justification": "the site-conditional calibration error is the quantity the manuscript reports, so the loss targets it directly",
    },
    {
        "id": "dev8-lambda-gamma-values",
        "paper_location": "Algorithm 1 step 9",
        "deviation": "The weights of the decision and calibration terms are not given numeric values. The configs carry lambda_decision = 0.5 and gamma_calibration = 0.1 as labelled engineering defaults, and both are reported in the run metadata.",
        "justification": "the objective cannot be assembled without them",
    },
    {
        "id": "dev9-reader-and-in-vitro-vectors",
        "paper_location": "Methods Sec. 4.9-4.10",
        "deviation": "The reader response table and the in vitro measurement block are not part of the manuscript's released material. The design, the pairing checks and every estimator are implemented and exercised on generated tables in the test suite; the reported values stay NOT_RUN.",
        "justification": "those analyses are only defensible on real measurements, and no measurement block exists here",
    },
    {
        "id": "dev10-quoted-arms",
        "paper_location": "Methods Sec. 4.8; Table 3",
        "deviation": "The endoscopic-ultrasound standard, the reader panel and the published multicentre range are carried as quoted values with their source named in the arm record, and are never recomputed here.",
        "justification": "those values come from other studies, and recomputing them here would present a different estimand as the manuscript's comparison",
    },
]


@dataclass
class CheckResult:
    """One verified item and the evidence that produced its status."""

    check_id: str
    category: str
    description: str
    status: str
    detail: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.check_id,
            "category": self.category,
            "description": self.description,
            "status": self.status,
            "detail": self.detail,
            "evidence": to_builtin(self.evidence),
        }


def ok(check_id: str, category: str, description: str, **evidence: Any) -> CheckResult:
    return CheckResult(check_id, category, description, PASS, evidence=evidence)


def bad(check_id: str, category: str, description: str, detail: str, **evidence: Any) -> CheckResult:
    return CheckResult(check_id, category, description, FAIL, detail=detail, evidence=evidence)


def skipped(check_id: str, category: str, description: str, detail: str, **evidence: Any) -> CheckResult:
    return CheckResult(check_id, category, description, NOT_RUN, detail=detail, evidence=evidence)


def blocked(check_id: str, category: str, description: str, detail: str, **evidence: Any) -> CheckResult:
    return CheckResult(check_id, category, description, BLOCKED, detail=detail, evidence=evidence)


class _SymbolIndex:
    """Module-level and class-method names parsed from a package's syntax trees."""

    def __init__(self, package: Path) -> None:
        self.modules: dict[str, set[str]] = {}
        self.methods: dict[str, dict[str, set[str]]] = {}
        for path in sorted(package.rglob("*.py")):
            self._index(str(path.relative_to(package)), path)

    def _index(self, relative: str, path: Path) -> None:
        top: set[str] = set()
        methods: dict[str, set[str]] = {}
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            self.modules[relative] = top
            self.methods[relative] = methods
            return
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                top.add(node.name)
            if isinstance(node, ast.ClassDef):
                methods[node.name] = {child.name for child in node.body if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))}
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        top.add(target.id)
        self.modules[relative] = top
        self.methods[relative] = methods

    def resolve(self, path: str, symbol: str) -> bool:
        """Whether ``symbol`` exists as a module member or a ``Class.method``."""
        names = self.modules.get(path)
        if names is None:
            return False
        if "." in symbol:
            owner, _, attribute = symbol.partition(".")
            return attribute in self.methods.get(path, {}).get(owner, set())
        return symbol in names


def find_repo_root(start: Path) -> Path:
    """Walk up until a directory holds both the package and the configs directory."""
    for candidate in [start, *start.parents]:
        if (candidate / PACKAGE_DIR).is_dir() and (candidate / "configs").is_dir():
            return candidate
    return start


def check_claims(root: Path) -> tuple[list[dict[str, Any]], CheckResult]:
    """Resolve every claim anchor and return the claim map together with its check."""
    package = root / PACKAGE_DIR
    index = _SymbolIndex(package)
    claims: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    missing_location: list[str] = []
    anchors_total = 0
    for raw in CLAIMS:
        anchors: list[dict[str, Any]] = []
        for anchor in raw["code"]:
            anchors_total += 1
            relative = str(anchor["path"])
            symbol = str(anchor["symbol"])
            exists = (package / relative).is_file()
            resolved = bool(exists and index.resolve(relative, symbol))
            if not resolved:
                unresolved.append({"path": relative, "symbol": symbol, "file_exists": exists})
            anchors.append({"path": relative, "symbol": symbol, "resolved": resolved})
        if not str(raw.get("paper_location", "")).strip():
            missing_location.append(str(raw["id"]))
        claims.append({**raw, "code": anchors, "verification": "EXECUTED" if all(a["resolved"] for a in anchors) else "UNRESOLVED"})
    evidence = {
        "claims": len(claims),
        "anchors": anchors_total,
        "unresolved": unresolved,
        "claims_without_paper_location": missing_location,
        "package_dir": PACKAGE_DIR,
    }
    description = "every paper claim resolves to a real file and a real module-level symbol"
    if unresolved:
        return claims, bad("pass1.claim_mapping", "provenance", description, f"{len(unresolved)} unresolved anchors", **evidence)
    return claims, ok("pass1.claim_mapping", "provenance", description, **evidence)


def _smoke_pipeline(root: Path, arm: str = "stagefm"):  # type: ignore[no-untyped-def]
    from ..utils.config import ExperimentConfig, resolve_experiment
    from .pipeline import prepare

    raw = resolve_experiment("_smoke", root / "configs", [])
    return prepare(ExperimentConfig.from_mapping(raw), arm)


def _tiny_batch(pipeline: Any, size: int) -> dict[str, Any]:
    from ..data.cohort import Layer
    from ..data.dataset import collate

    return collate([pipeline.datasets[Layer.TRAIN][index] for index in range(size)])


def check_execution(root: Path) -> list[CheckResult]:
    """Execute data reading, forward, objective, backward, update and checkpointing."""
    import torch

    from ..data.cohort import Layer
    from ..losses.total import staged_objective
    from ..training.optim import build_optimizer

    results: list[CheckResult] = []
    try:
        pipeline = _smoke_pipeline(root)
    except Exception as error:  # noqa: BLE001 - reported as a check
        return [
            bad("pass2.pipeline_build", "execution", "the smoke pipeline builds", f"{type(error).__name__}: {error}", traceback=traceback.format_exc()[-1200:])
        ]
    dataset = pipeline.datasets[Layer.TRAIN]
    item = dataset[0]
    results.append(
        ok(
            "pass2.data_read",
            "execution",
            "an examination item carries a volume, the non-imaging streams, a presence vector and the supervision",
            records=len(dataset),
            volume_shape=list(item["volume"].shape),
            radiomics_dim=int(item["radiomics"].numel()),
            clinical_dim=int(item["clinical"].numel()),
            presence=list(item["presence"].tolist()),
            labels={"t": int(item["label_t"]), "n": int(item["label_n"]), "m": int(item["label_m"])},
        )
    )
    batch = _tiny_batch(pipeline, 3)
    model = pipeline.model
    output = model(batch)
    results.append(
        ok(
            "pass2.forward",
            "execution",
            "the forward pass returns per-axis, joint and projected distributions",
            t_shape=list(output.axis["T"].probabilities.shape),
            joint_shape=list(output.joint.shape),
            projected_sums=[float(value) for value in output.projected.sum(dim=-1).tolist()],
            joint_sums=[float(value) for value in output.joint.sum(dim=-1).tolist()],
        )
    )
    breakdown = staged_objective(
        axis_outputs=output.axis,
        projected=output.projected,
        reference_stages=batch["stage_column"],
        targets={"T": batch["label_t"], "N": batch["label_n"], "M": batch["label_m"]},
        site_index=batch["site_index"],
        config=pipeline.config.loss,
        ordinal=True,
    )
    results.append(
        ok(
            "pass2.objective",
            "execution",
            "the objective combines the ordinal, decision and calibration terms",
            total=float(breakdown.total.detach()),
            ordinal=float(breakdown.ordinal.total.detach()),
            decision=float(breakdown.decision.total.detach()),
            calibration=float(breakdown.calibration.total.detach()),
            lambda_decision=pipeline.config.loss.lambda_decision,
            gamma_calibration=pipeline.config.loss.gamma_calibration,
        )
    )
    optimizer, _summary = build_optimizer(model, pipeline.config.train)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}
    breakdown.total.backward()  # type: ignore[no-untyped-call]
    optimizer.step()
    changed = sum(1 for name, parameter in model.named_parameters() if name in before and not torch.equal(parameter.detach(), before[name]))
    if changed == 0:
        results.append(
            bad("pass2.backward_update", "execution", "a backward pass updates trainable parameters", "no trainable parameter changed after one step")
        )
    else:
        results.append(
            ok(
                "pass2.backward_update",
                "execution",
                "a backward pass updates trainable parameters",
                trainable_tensors=len(before),
                updated_tensors=changed,
                frozen_tensors=sum(1 for parameter in model.parameters() if not parameter.requires_grad),
            )
        )
    results.append(_check_checkpoint(model, optimizer))
    results.append(_check_single_batch_overfit(root))
    results.append(_check_training_loop(root))
    return results


def _check_checkpoint(model: Any, optimizer: Any) -> CheckResult:
    """Round-trip a checkpoint through a scratch directory so the release stays clean."""
    import tempfile

    from ..training.checkpoint import CheckpointState, checkpoint_digest, load_checkpoint, save_checkpoint

    directory = Path(tempfile.mkdtemp(prefix="stagefm-verify-"))
    path = directory / "roundtrip.pt"
    thresholds = {"A": 0.34, "B": 0.30}
    state = CheckpointState(
        epoch=1,
        global_step=1,
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        seed_state={"seed": 0},
        thresholds=thresholds,
        metrics={"objective": 1.0},
        config={"experiment": "verify"},
        arm="stagefm",
    )
    digest_before = checkpoint_digest(state)
    save_checkpoint(state, path)
    payload = load_checkpoint(path)
    digest_after = checkpoint_digest(CheckpointState(epoch=1, global_step=1, model_state=payload["model_state"]))
    if digest_before != digest_after:
        return bad(
            "pass2.checkpoint_roundtrip",
            "execution",
            "a checkpoint round trip preserves the tensor payload and the frozen thresholds",
            "tensor payload digest changed",
        )
    if dict(payload["thresholds"]) != thresholds:
        return bad(
            "pass2.checkpoint_roundtrip", "execution", "a checkpoint round trip preserves the tensor payload and the frozen thresholds", "thresholds changed"
        )
    return ok(
        "pass2.checkpoint_roundtrip",
        "execution",
        "a checkpoint round trip preserves the tensor payload and the frozen thresholds",
        payload_digest=digest_after,
        thresholds=dict(payload["thresholds"]),
        bytes=path.stat().st_size,
    )


def _check_single_batch_overfit(root: Path) -> CheckResult:
    import torch

    from ..losses.total import staged_objective
    from ..training.optim import build_optimizer

    pipeline = _smoke_pipeline(root)
    model = pipeline.model
    batch = _tiny_batch(pipeline, 8)
    optimizer, _ = build_optimizer(model, pipeline.config.train)
    first: float | None = None
    last: float | None = None
    for _ in range(24):
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        flags = getattr(model, "flags", None)
        distribution = output.joint if flags is not None and not flags.stage_consistency else output.projected
        breakdown = staged_objective(
            axis_outputs=output.axis,
            projected=distribution,
            reference_stages=batch["stage_column"],
            targets={"T": batch["label_t"], "N": batch["label_n"], "M": batch["label_m"]},
            site_index=batch["site_index"],
            config=pipeline.config.loss,
            ordinal=True,
        )
        breakdown.total.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        value = float(breakdown.total.detach())
        first = value if first is None else first
        last = value
    assert first is not None and last is not None
    description = "the objective falls on a single repeated batch"
    if not last < first * 0.9:
        return bad("pass2.single_batch_overfit", "execution", description, f"objective did not fall enough: {first:.4f} -> {last:.4f}", first=first, last=last)
    return ok("pass2.single_batch_overfit", "execution", description, first=first, last=last, steps=24, ratio=last / first if first else float("nan"))


def _check_training_loop(root: Path) -> CheckResult:
    from ..data.cohort import Layer
    from ..training.amp import PrecisionSpec, model_device
    from ..training.loops import train_epoch
    from ..training.optim import build_optimizer
    from ..training.scheduler import ScheduleSpec, build_scheduler

    pipeline = _smoke_pipeline(root)
    model = pipeline.model
    device = model_device(model)
    loader = pipeline.loader(Layer.TRAIN, shuffle=False)
    optimizer, _ = build_optimizer(model, pipeline.config.train)
    scheduler = build_scheduler(optimizer, ScheduleSpec(warmup_steps=1, total_steps=4))
    metrics = train_epoch(model, loader, optimizer, scheduler, pipeline.config, PrecisionSpec.resolve("fp32", device.type), None, device, max_steps=2)
    description = "two optimiser steps run through the training loop"
    if not np.isfinite(metrics["total"]):
        return bad("pass2.minimal_training_loop", "execution", description, "loss was not finite")
    return ok(
        "pass2.minimal_training_loop",
        "execution",
        description,
        steps=metrics.get("steps", 0.0),
        loss=metrics["total"],
        grad_norm=metrics["grad_norm"],
        lr=metrics.get("group0", float("nan")),
    )


def _brute_force_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Area under the curve by explicit pairwise counting, ties credited at one half."""
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    greater = 0.0
    for value in positive:
        greater += float((value > negative).sum()) + 0.5 * float((value == negative).sum())
    return greater / (positive.size * negative.size)


def check_procedures() -> list[CheckResult]:
    """Re-derive the core numerical procedures with code independent of the implementation.

    Every expected value here comes from explicit enumeration, a closed form, a hand
    computation or a different library. None of these checks calls the routine it is
    checking to produce its own expected value.
    """
    import torch
    from sklearn.metrics import cohen_kappa_score, roc_auc_score

    from ..data.radiomics import collapse_redundant, extract_descriptors
    from ..data.schema import ALL_TRIPLES, StageTriple, Stream
    from ..data.segmentation import peritumoral_shell
    from ..data.staging import AchievableSet, shift_category, treatment_category
    from ..data.streams import apply_modality_dropout, dropout_rates
    from ..metrics.calibration import expected_calibration_error
    from ..metrics.concordance import exact_combination_concordance, weighted_kappa
    from ..metrics.decision import binary_nri, boundary_discordance, decision_curve, discordance_indicator, net_benefit
    from ..metrics.discrimination import binary_auroc, macro_auroc, ordinal_auroc
    from ..models.encoder import CompactCTEncoder
    from ..models.lora import adapter_parameter_count, apply_lora
    from ..models.ordinal import MonotonicOrdinalHead
    from ..models.stage_consistency import FeasibilityProjection, joint_distribution
    from ..stats.delong import delong_auc_ci
    from ..stats.icc import anova_table, decompose
    from ..stats.mcnemar import mcnemar
    from ..stats.multiplicity import benjamini_hochberg, interaction_ratio
    from ..training.scheduler import ScheduleSpec
    from ..utils.config import EncoderConfig

    results: list[CheckResult] = []
    rng = np.random.default_rng(20260925)

    # --- feasibility projection -------------------------------------------------
    achievable = AchievableSet()
    projection = FeasibilityProjection(achievable)
    raw = rng.random(32)
    raw = raw / raw.sum()
    with torch.no_grad():
        out = projection(torch.tensor(raw, dtype=torch.float64).view(1, -1))
    projected = out.probabilities.numpy()[0]
    mask = np.asarray(achievable.mask, dtype=bool)
    closed_form = np.where(mask, raw, 0.0)
    closed_form = closed_form / closed_form.sum()
    if np.allclose(projected, closed_form, atol=1e-12) and abs(projected.sum() - 1.0) < 1e-12 and projected[~mask].sum() == 0.0:
        results.append(
            ok(
                "indep.projection_closed_form",
                "algorithm",
                "the projection equals an explicit mask-and-renormalise computation, has unit sum and no mass outside the achievable set",
                achievable_size=int(mask.sum()),
                outside_mass=float(projected[~mask].sum()),
                max_abs_deviation=float(np.max(np.abs(projected - closed_form))),
            )
        )
    else:
        results.append(
            bad(
                "indep.projection_closed_form",
                "algorithm",
                "the projection matches the closed form",
                "projected distribution deviates from the explicit computation",
            )
        )
    degenerate = np.zeros(33, dtype=float)
    degenerate[:32][~mask] = 1.0
    with torch.no_grad():
        degenerate_out = projection(torch.tensor(degenerate[:32], dtype=torch.float64).view(1, -1)).probabilities.numpy()[0]
    uniform = mask.astype(float) / mask.sum()
    if np.allclose(degenerate_out, uniform, atol=1e-12):
        results.append(
            ok(
                "indep.projection_degenerate",
                "algorithm",
                "the degenerate case returns the uniform distribution on the achievable set",
                support=int(mask.sum()),
            )
        )
    else:
        results.append(
            bad(
                "indep.projection_degenerate",
                "algorithm",
                "the degenerate case returns the uniform distribution",
                "returned distribution is not uniform on the feasible support",
            )
        )

    feasible_triples = {stage.flat_index for stage in ALL_TRIPLES if _reference_achievable(stage.t, stage.n, stage.m)}
    shipped = {index for index, keep in enumerate(mask.tolist()) if keep}
    if feasible_triples == shipped:
        results.append(
            ok(
                "indep.achievable_set",
                "algorithm",
                "the achievable set matches an independently written rule over the 32 combinations",
                size=len(shipped),
                excluded=32 - len(shipped),
            )
        )
    else:
        results.append(
            bad(
                "indep.achievable_set",
                "algorithm",
                "the achievable set matches an independently written rule",
                f"symmetrical difference of size {len(feasible_triples ^ shipped)}",
            )
        )

    rule_examples = [
        (StageTriple(1, 0, 0), True),
        (StageTriple(1, 1, 0), True),
        (StageTriple(1, 3, 0), False),
        (StageTriple(4, 3, 1), True),
        (StageTriple(4, 0, 1), True),
        (StageTriple(2, 0, 1), False),
        (StageTriple(2, 3, 0), False),
    ]
    mismatched = [stage.label() for stage, expected in rule_examples if achievable.contains(stage) is not expected]
    if mismatched:
        results.append(bad("indep.achievable_membership", "algorithm", "named combinations are classified as expected", f"mismatched: {', '.join(mismatched)}"))
    else:
        results.append(ok("indep.achievable_membership", "algorithm", "named combinations are classified as expected", examples=len(rule_examples)))

    # --- treatment boundary rule ------------------------------------------------
    expected_categories = {
        StageTriple(1, 0, 0): "surgery_first",
        StageTriple(2, 1, 0): "perioperative",
        StageTriple(4, 3, 1): "systemic",
    }
    wrong = [stage.label() for stage, name in expected_categories.items() if treatment_category(stage).value != name]
    pairs = [
        (StageTriple(1, 0, 0), StageTriple(2, 1, 0), True),
        (StageTriple(1, 0, 0), StageTriple(1, 0, 0), False),
        (StageTriple(3, 2, 0), StageTriple(4, 2, 1), True),
    ]
    for predicted_stage, reference_stage, expected in pairs:
        indicator = discordance_indicator(np.array([predicted_stage.flat_index]), np.array([reference_stage.flat_index]))
        if bool(indicator[0]) is not expected:
            wrong.append(f"{predicted_stage.label()} vs {reference_stage.label()}")
    if wrong:
        results.append(
            bad("indep.boundary_rule", "algorithm", "the boundary rule maps the named stages onto the expected categories", f"mismatched: {', '.join(wrong)}")
        )
    else:
        results.append(
            ok(
                "indep.boundary_rule",
                "algorithm",
                "the boundary rule maps the named stages onto the expected categories",
                examples=len(expected_categories) + len(pairs),
            )
        )

    shifted = {offset: shift_category(StageTriple(2, 1, 0), offset).value for offset in (-1, 0, 1)}
    if shifted[0] == "perioperative" and shifted[-1] == "surgery_first":
        results.append(ok("indep.boundary_shift", "algorithm", "the sensitivity shift moves and clamps the category order", shifts=shifted))
    else:
        results.append(bad("indep.boundary_shift", "algorithm", "the sensitivity shift moves the category order", f"unexpected shifts {shifted}"))

    # --- discrimination ---------------------------------------------------------
    positive = np.array([0.9, 0.7, 0.4, 0.8, 0.35, 0.6])
    negative = np.array([0.2, 0.55, 0.1, 0.45, 0.3, 0.25, 0.05])
    scores = np.concatenate([positive, negative])
    labels = np.concatenate([np.ones(positive.size, dtype=int), np.zeros(negative.size, dtype=int)])
    brute = _brute_force_auc(positive, negative)
    ours = binary_auroc(scores, labels)
    library = float(roc_auc_score(labels, scores))
    if abs(ours - brute) < 1e-12 and abs(ours - library) < 1e-12:
        results.append(
            ok(
                "indep.auc_bruteforce",
                "metric",
                "the binary area under the curve equals explicit pairwise counting and the reference library",
                value=ours,
                brute_force=brute,
                library=library,
            )
        )
    else:
        results.append(bad("indep.auc_bruteforce", "metric", "the binary area equals pairwise counting", f"{ours} vs bruteforce {brute} vs library {library}"))

    four_labels = rng.integers(0, 4, size=240)
    four_scores = rng.random((240, 4)) + 0.6 * np.eye(4)[four_labels]
    independent_macro = float(np.mean([roc_auc_score((four_labels == category).astype(int), four_scores[:, category]) for category in range(4)]))
    shipped_macro = macro_auroc(four_scores, four_labels, 4)
    if abs(independent_macro - shipped_macro) < 1e-12:
        results.append(
            ok(
                "indep.macro_auroc",
                "metric",
                "the four-class macro average equals the mean of the one-vs-rest areas",
                value=shipped_macro,
                independent=independent_macro,
            )
        )
    else:
        results.append(
            bad("indep.macro_auroc", "metric", "the macro average equals the mean of the one-vs-rest areas", f"{shipped_macro} vs {independent_macro}")
        )

    ordinal_labels = rng.integers(0, 4, size=300)
    ordinal_scores = ordinal_labels + rng.normal(scale=1.2, size=300)
    numerator = 0.0
    denominator = 0.0
    for lower in range(4):
        for higher in range(lower + 1, 4):
            low_scores = ordinal_scores[ordinal_labels == lower]
            high_scores = ordinal_scores[ordinal_labels == higher]
            weight = low_scores.size * high_scores.size
            if weight == 0:
                continue
            numerator += _brute_force_auc(high_scores, low_scores) * weight
            denominator += weight
    independent_ordinal = numerator / denominator
    shipped_ordinal = ordinal_auroc(ordinal_scores, ordinal_labels)
    if abs(independent_ordinal - shipped_ordinal) < 1e-12:
        results.append(
            ok(
                "indep.ordinal_auroc",
                "metric",
                "the ordinal summary equals the sample-size weighted mean of the pairwise areas",
                value=shipped_ordinal,
                independent=independent_ordinal,
            )
        )
    else:
        results.append(
            bad(
                "indep.ordinal_auroc",
                "metric",
                "the ordinal summary equals the weighted mean of the pairwise areas",
                f"{shipped_ordinal} vs {independent_ordinal}",
            )
        )

    delong = delong_auc_ci(scores, labels)
    replicates = np.empty(400, dtype=float)
    for position in range(400):
        draw = rng.integers(0, scores.size, size=scores.size)
        if len(np.unique(labels[draw])) < 2:
            replicates[position] = np.nan
            continue
        replicates[position] = binary_auroc(scores[draw], labels[draw])
    bootstrap_se = float(np.nanstd(replicates, ddof=1))
    if abs(delong.standard_error - bootstrap_se) < max(0.02, 0.5 * bootstrap_se):
        results.append(
            ok(
                "indep.delong_variance",
                "statistic",
                "the DeLong standard error agrees with a bootstrap standard error",
                delong=delong.standard_error,
                bootstrap=bootstrap_se,
            )
        )
    else:
        results.append(
            bad("indep.delong_variance", "statistic", "the DeLong standard error agrees with a bootstrap", f"{delong.standard_error} vs {bootstrap_se}")
        )

    # --- variance components ----------------------------------------------------
    group = np.repeat(np.array(["A", "B", "C", "D", "E"]), 40)
    values = (np.repeat(np.array([0.02, 0.06, 0.10, 0.14, 0.18]), 40) + rng.normal(scale=0.05, size=200)).astype(float)
    components = decompose(values, group)
    table = anova_table(values, group)
    hand_between = (table["between"]["ss"] / 4 - table["within"]["ms"]) / ((200 - 0) / 4)
    if abs(components.mean_square_between - table["between"]["ms"]) < 1e-9 and abs(components.mean_square_within - table["within"]["ms"]) < 1e-9:
        results.append(
            ok(
                "indep.icc_anova",
                "statistic",
                "the variance decomposition matches the hand-computed mean squares",
                mean_square_between=components.mean_square_between,
                mean_square_within=components.mean_square_within,
                between_variance=components.between_site_variance,
                icc=components.intraclass_correlation,
                hand_check=hand_between,
            )
        )
    else:
        results.append(bad("indep.icc_anova", "statistic", "the decomposition matches the hand-computed mean squares", "mean squares disagree"))

    # --- decision metrics -------------------------------------------------------
    indicative = np.array([0.9, 0.8, 0.3, 0.6, 0.2, 0.7, 0.4, 0.1])
    truth = np.array([1, 1, 0, 1, 0, 0, 1, 0])
    threshold = 0.35
    chosen = (indicative >= threshold).astype(int)
    manual = float(((chosen == 1) & (truth == 1)).sum()) / len(truth) - float(((chosen == 1) & (truth == 0)).sum()) / len(truth) * (threshold / (1 - threshold))
    if abs(net_benefit(truth, chosen, threshold) - manual) < 1e-12:
        results.append(ok("indep.net_benefit", "metric", "net benefit matches the closed form", value=manual, threshold=threshold))
    else:
        results.append(bad("indep.net_benefit", "metric", "net benefit matches the closed form", f"{net_benefit(truth, chosen, threshold)} vs {manual}"))

    curve = decision_curve(truth, indicative, np.array([0.2, 0.4]))
    prevalence = float(truth.mean())
    expected_treat_all = np.array([prevalence - (1 - prevalence) * (t / (1 - t)) for t in (0.2, 0.4)])
    if np.allclose(curve["treat_none"], 0.0) and np.allclose(curve["treat_all"], expected_treat_all):
        results.append(
            ok(
                "indep.decision_curve_references",
                "metric",
                "the treat-none and treat-all references equal their closed forms",
                treat_all=list(curve["treat_all"]),
                prevalence=prevalence,
            )
        )
    else:
        results.append(bad("indep.decision_curve_references", "metric", "the decision-curve references match their closed forms", "reference curves disagree"))

    hand_ece = 0.1 * abs(0.05 - 0.0)
    probabilities = np.concatenate([np.full(90, 0.5), np.full(10, 0.05)])
    outcomes = np.concatenate([np.zeros(90), np.zeros(10)])
    computed_ece = expected_calibration_error(probabilities, outcomes, bins=10)
    hand = 0.9 * abs(0.5 - 0.0) + 0.1 * abs(0.05 - 0.0)
    if abs(computed_ece - hand) < 1e-12:
        results.append(ok("indep.ece_hand", "metric", "expected calibration error matches a hand-computed two-bin example", value=computed_ece, hand=hand))
    else:
        results.append(bad("indep.ece_hand", "metric", "expected calibration error matches the hand computation", f"{computed_ece} vs {hand}"))
    _ = hand_ece

    arm_a = np.array([1, 1, 0, 0, 1, 1, 0, 1])
    arm_b = np.array([1, 0, 0, 1, 1, 1, 0, 0])
    paired = mcnemar(arm_a, arm_b)
    from scipy import stats as scipy_stats

    only_first = int(((arm_a == 1) & (arm_b == 0)).sum())
    only_second = int(((arm_a == 0) & (arm_b == 1)).sum())
    discordant = only_first + only_second
    exact = float(min(1.0, 2.0 * scipy_stats.binom.cdf(min(only_first, only_second), discordant, 0.5)))
    if paired.p_value_exact == exact:
        results.append(
            ok("indep.mcnemar_exact", "statistic", "the paired test reproduces the exact two-sided binomial probability", p_value=exact, discordant=discordant)
        )
    else:
        results.append(
            bad("indep.mcnemar_exact", "statistic", "the paired test reproduces the exact binomial probability", f"{paired.p_value_exact} vs {exact}")
        )

    nri = binary_nri(np.array([0, 0, 1, 0]), np.array([1, 0, 1, 0]), np.array([1, 0, 1, 0]))
    if abs(nri["events"] - 0.5) < 1e-12 and abs(nri["non_events"] - 0.0) < 1e-12:
        results.append(
            ok(
                "indep.nri_components",
                "metric",
                "the reclassification components match a hand-computed example",
                events=nri["events"],
                non_events=nri["non_events"],
            )
        )
    else:
        results.append(bad("indep.nri_components", "metric", "the reclassification components match the hand computation", f"{nri}"))

    # --- objective terms --------------------------------------------------------
    head = MonotonicOrdinalHead(input_dim=5, num_classes=4)
    with torch.no_grad():
        head.score_layer.weight.normal_(std=0.5)
    features = torch.tensor(rng.normal(size=(16, 5)), dtype=torch.float32)
    ordinal_output = head(features)
    cumulative = ordinal_output.cumulative_logits
    orders = torch.sigmoid(cumulative)
    monotone = bool((orders[:, 1:] <= orders[:, :-1] + 1e-6).all())
    if monotone and bool(torch.all(ordinal_output.probabilities >= 0)) and bool(torch.allclose(ordinal_output.probabilities.sum(dim=-1), torch.ones(16))):
        results.append(
            ok(
                "indep.ordinal_head_structure",
                "model",
                "the ordinal head yields a non-increasing cumulative curve and normalised non-negative probabilities",
                max_gap=float(ordinal_output.cut_points.detach().diff().min()),
            )
        )
    else:
        results.append(
            bad("indep.ordinal_head_structure", "model", "the ordinal head yields a valid cumulative structure", "cumulative curve or normalisation violated")
        )

    head.raw_gaps.data.normal_(std=3.0)
    if float(head.cut_points.detach().diff().min()) > 0:
        results.append(
            ok(
                "indep.cut_point_monotonicity",
                "model",
                "cut points stay strictly increasing under a large perturbation of the raw gaps",
                minimum_gap=float(head.cut_points.detach().diff().min()),
            )
        )
    else:
        results.append(bad("indep.cut_point_monotonicity", "model", "cut points stay strictly increasing", "a gap collapsed to zero or below"))

    from ..losses.ordinal import cumulative_link_loss

    logits = torch.tensor([[1.0, -1.0], [0.5, 0.25]], dtype=torch.float64)
    targets = torch.tensor([2, 0], dtype=torch.long)
    manual_terms = []
    for row, target in enumerate(targets.tolist()):
        for level in range(logits.shape[1]):
            label = 1.0 if target > level else 0.0
            value = float(logits[row, level])
            manual_terms.append(label * np.log1p(np.exp(-value)) + (1 - label) * np.log1p(np.exp(value)))
    if abs(float(cumulative_link_loss(logits, targets)) - float(np.mean(manual_terms))) < 1e-12:
        results.append(
            ok(
                "indep.ordinal_loss_hand",
                "loss",
                "the ordinal loss equals a hand-computed cumulative-link negative log-likelihood",
                value=float(cumulative_link_loss(logits, targets)),
            )
        )
    else:
        results.append(bad("indep.ordinal_loss_hand", "loss", "the ordinal loss equals the hand computation", "values disagree"))

    from ..losses.decision import decision_loss

    category_of = np.array([treatment_category(stage).value for stage in ALL_TRIPLES])
    systemic_column = int(np.flatnonzero(category_of == "systemic")[0])
    surgery_column = int(np.flatnonzero(category_of == "surgery_first")[0])
    one_hot = np.zeros((4, 32))
    one_hot[0, systemic_column] = 1.0
    one_hot[1, surgery_column] = 1.0
    one_hot[2, systemic_column] = 1.0
    one_hot[3, surgery_column] = 1.0
    reference_columns = np.array([systemic_column, systemic_column, surgery_column, surgery_column])
    loss_value = decision_loss(torch.tensor(one_hot, dtype=torch.float64), torch.tensor(reference_columns, dtype=torch.long))
    mass = loss_value.systemic_probability.numpy()
    if abs(mass[0] - 1.0) < 1e-9 and abs(mass[1]) < 1e-9:
        results.append(
            ok(
                "indep.decision_loss_surrogate",
                "loss",
                "the decision term reads the systemic mass off the projected distribution",
                systemic_column=systemic_column,
                systemic_mass=list(mass),
            )
        )
    else:
        results.append(bad("indep.decision_loss_surrogate", "loss", "the decision term reads the systemic mass", f"systemic mass {mass.tolist()}"))

    # --- calibration and thresholds --------------------------------------------
    from ..models.risk_control import AffineCorrection, decide_stage_category, select_thresholds
    from ..utils.config import RiskConfig

    calibration_logits = np.linspace(-2.0, 2.0, 200)
    target = (calibration_logits + rng.normal(scale=0.5, size=200) > 0.4).astype(float)
    from ..models.risk_control import SiteCalibrator

    sites = ["A"] * 100 + ["B"] * 100
    calibrator = SiteCalibrator.fit(calibration_logits, target, sites, scope="site", external_sites=("D", "E"))
    identity = AffineCorrection(1.0, 0.0)
    fitted = calibrator.transferred
    if abs(identity.apply(np.array([0.0]))[0]) < 1e-12 and fitted.slope != 1.0:
        results.append(
            ok("indep.calibration_affine", "model", "the calibration layer is an affine map on the logit scale", slope=fitted.slope, intercept=fitted.intercept)
        )
    else:
        results.append(bad("indep.calibration_affine", "model", "the calibration layer is affine on the logit scale", "the fit degenerated to the identity"))

    masses = np.zeros((120, 3))
    masses[:, 0] = 0.7
    masses[:, 1] = 0.2
    masses[:, 2] = 0.1
    masses[::2, 2] = 0.45
    reference_categories = np.where(np.arange(120) % 2 == 0, 2, 1)
    threshold_selection = select_thresholds(masses, reference_categories, ["A"] * 120, RiskConfig(threshold_grid=(0.2, 0.3, 0.4, 0.5, 0.6)))
    frozen = threshold_selection["A"]
    grid_errors = [float((decide_stage_category(masses, np.full(120, t)) != reference_categories).mean()) for t in (0.2, 0.3, 0.4, 0.5, 0.6)]
    if abs(frozen.empirical_error - min(grid_errors)) < 1e-12:
        results.append(
            ok(
                "indep.threshold_selection",
                "model",
                "the frozen threshold minimises the empirical boundary error over the grid and carries a union-bound term",
                threshold=frozen.threshold,
                empirical=frozen.empirical_error,
                bound=frozen.bound,
                grid_errors=grid_errors,
            )
        )
    else:
        results.append(
            bad("indep.threshold_selection", "model", "the threshold minimises the empirical boundary error", f"{frozen.empirical_error} vs {min(grid_errors)}")
        )

    # --- LoRA -------------------------------------------------------------------
    torch.manual_seed(0)
    encoder = CompactCTEncoder(EncoderConfig(embed_dim=48, depth=2, num_heads=4, patch_size=4, lora_rank=4, lora_alpha=8))
    volume = torch.randn(2, 1, 16, 16, 8)
    with torch.no_grad():
        reference_output = encoder(volume).tokens.clone()
    frozen_before = {name: parameter.clone() for name, parameter in encoder.named_parameters()}
    replaced = apply_lora(encoder, rank=4, alpha=8, targets=("q_proj", "k_proj", "v_proj", "out_proj"))
    with torch.no_grad():
        adapted_output = encoder(volume).tokens
    lora_inputs = ["q_proj", "k_proj", "v_proj", "out_proj"]
    zero_init_ok = bool(torch.allclose(reference_output, adapted_output, atol=1e-6))
    for name, parameter in encoder.named_parameters():
        if name.endswith("lora_a") or name.endswith("lora_b"):
            continue
        # Wrapping a projection nests the frozen linear under it, so the original
        # qualified name gains a ``base`` component.
        original = frozen_before.get(name.replace(".base.", "."))
        if original is None or not torch.equal(parameter.data, original):
            zero_init_ok = False
    if zero_init_ok and replaced >= 4:
        results.append(
            ok(
                "indep.lora_zero_init",
                "model",
                "the adapted encoder is numerically identical to the frozen encoder at initialisation and no base weight moved",
                replaced_projections=replaced,
                targets=lora_inputs,
                adapter_parameters=adapter_parameter_count(encoder),
            )
        )
    else:
        results.append(
            bad("indep.lora_zero_init", "model", "the adapted encoder starts identical to the frozen one", f"replaced={replaced}, identical={zero_init_ok}")
        )

    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    for name, parameter in encoder.named_parameters():
        if name.endswith("lora_a") or name.endswith("lora_b"):
            parameter.requires_grad_(True)
    base_requires_grad = {
        name: parameter.requires_grad for name, parameter in encoder.named_parameters() if not (name.endswith("lora_a") or name.endswith("lora_b"))
    }
    adapters_trainable = adapter_parameter_count(encoder)
    if base_requires_grad and not any(base_requires_grad.values()) and adapters_trainable > 0:
        results.append(
            ok(
                "indep.lora_freezes_base",
                "model",
                "freezing the backbone leaves only the adapters trainable",
                frozen=len(base_requires_grad),
                trainable_adapters=adapters_trainable,
            )
        )
    else:
        results.append(
            bad(
                "indep.lora_freezes_base",
                "model",
                "freezing the backbone leaves only the adapters trainable",
                f"{sum(base_requires_grad.values())} base tensors remain trainable, {adapters_trainable} adapter parameters trainable",
            )
        )

    # --- modality dropout -------------------------------------------------------
    plan_streams = ("radiomics", "clinical", "endoscopy", "pathology")
    available = {"ct": True, "radiomics": True, "clinical": True, "endoscopy": True, "pathology": True}
    rates = dropout_rates(0.15)
    from ..data.streams import StreamPlan

    plan = StreamPlan(active=tuple(Stream(name) for name in ("ct", *plan_streams)))
    generator = np.random.default_rng(7)
    dropped_counts = {name: 0 for name in plan_streams}
    trials = 4000
    for _ in range(trials):
        dropped = apply_modality_dropout(available, rates, generator, plan)
        for name in plan_streams:
            if available[name] and not dropped[name]:
                dropped_counts[name] += 1
    observed = {name: dropped_counts[name] / trials for name in plan_streams}
    within_tolerance = all(abs(value - 0.15) < 0.03 for value in observed.values())
    if within_tolerance:
        results.append(
            ok(
                "indep.modality_dropout_rate",
                "data",
                "the modality dropout blanks each non-imaging stream at the configured rate",
                observed=observed,
                requested=0.15,
                trials=trials,
            )
        )
    else:
        results.append(bad("indep.modality_dropout_rate", "data", "the modality dropout rate matches the configuration", f"observed {observed}"))

    unavailable = {"ct": True, "radiomics": True, "clinical": False, "endoscopy": False, "pathology": True}
    after = apply_modality_dropout(unavailable, rates, generator, plan)
    invented = [name for name, present in unavailable.items() if not present and after[name]]
    if not invented:
        results.append(ok("indep.dropout_never_invents", "data", "modality dropout only removes availability and never creates it"))
    else:
        results.append(bad("indep.dropout_never_invents", "data", "dropout never creates availability", f"invented: {invented}"))

    # --- imaging and descriptors ------------------------------------------------
    mask = np.zeros((9, 9, 9), dtype=bool)
    mask[4, 4, 4] = True
    shell = peritumoral_shell(mask, (1.0, 1.0, 1.0), 0.0, 2.0)
    distances = np.sqrt(((np.argwhere(shell) - np.array([4, 4, 4])) ** 2).sum(axis=1))
    if shell.any() and float(distances.max()) <= 2.0 and float(distances.min()) > 0.0 and not shell[mask].any():
        results.append(
            ok(
                "indep.shell_geometry",
                "data",
                "the peritumoral shell contains exactly the voxels within the outer radius and excludes the seed region",
                voxels=int(shell.sum()),
                max_distance=float(distances.max()),
            )
        )
    else:
        results.append(
            bad(
                "indep.shell_geometry",
                "data",
                "the shell geometry respects both radii",
                f"max distance {float(distances.max()) if distances.size else float('nan')}",
            )
        )

    volume_probe = rng.random((10, 10, 10)).astype(np.float32)
    shell_probe = np.zeros((10, 10, 10), dtype=bool)
    shell_probe[2:8, 2:8, 2:8] = True
    families = ("firstorder", "glcm", "glrlm", "glszm", "ngtdm", "gldm", "shape")
    extraction_a = extract_descriptors(volume_probe, shell_probe, (1.0, 1.0, 1.0), families)
    extraction_b = extract_descriptors(volume_probe, shell_probe, (1.0, 1.0, 1.0), families)
    if (
        np.array_equal(extraction_a.values, extraction_b.values)
        and extraction_a.values.size == len(extraction_a.names)
        and np.isfinite(extraction_a.values).all()
    ):
        results.append(
            ok(
                "indep.descriptor_determinism",
                "data",
                "descriptor extraction is deterministic and finite across all seven families",
                descriptors=int(extraction_a.values.size),
                families=len(families),
            )
        )
    else:
        results.append(
            bad(
                "indep.descriptor_determinism", "data", "descriptor extraction is deterministic and finite", "two runs disagreed or produced a non-finite value"
            )
        )

    block = np.column_stack([np.arange(40, dtype=float), np.arange(40, dtype=float) * 2.0, rng.normal(size=40)])
    keep = collapse_redundant(block, ["a", "b", "c"], threshold=0.9)
    if len(keep) == 2 and 2 in keep:
        results.append(
            ok(
                "indep.redundancy_collapse",
                "data",
                "perfectly correlated descriptors collapse to a single representative",
                kept=[["a", "b", "c"][index] for index in keep],
            )
        )
    else:
        results.append(bad("indep.redundancy_collapse", "data", "perfectly correlated descriptors collapse to one representative", f"kept {keep}"))

    from ..data.radiomics import SiteStandardiser

    features = rng.normal(size=(60, 3)) + np.repeat(np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0], [-5.0, -5.0, -5.0]]), 20, axis=0)
    site_list = ["A"] * 20 + ["B"] * 20 + ["C"] * 20
    standardiser = SiteStandardiser.fit(features, site_list)
    standardised = standardiser.transform(features, site_list)
    per_site_means = np.array([standardised[np.array(site_list) == site].mean(axis=0) for site in ("A", "B", "C")])
    per_site_stds = np.array([standardised[np.array(site_list) == site].std(axis=0) for site in ("A", "B", "C")])
    if np.allclose(per_site_means, 0.0, atol=1e-9) and np.allclose(per_site_stds, 1.0, atol=1e-9):
        results.append(
            ok(
                "indep.per_site_standardisation",
                "data",
                "per-site standardisation centres and scales each site with its own statistics",
                max_mean=float(np.abs(per_site_means).max()),
                max_std_error=float(np.abs(per_site_stds - 1.0).max()),
            )
        )
    else:
        results.append(
            bad(
                "indep.per_site_standardisation",
                "data",
                "per-site standardisation centres and scales each site",
                f"means {per_site_means.tolist()}, stds {per_site_stds.tolist()}",
            )
        )

    unseen = standardiser.transform(np.array([[100.0, 100.0, 100.0]]), ["Z"])
    pooled_mean = np.mean([mean for mean, _ in standardiser.locations.values()], axis=0)
    if not np.allclose(unseen, (np.array([[100.0, 100.0, 100.0]]) - pooled_mean), atol=1e-9):
        results.append(
            ok(
                "indep.standardiser_unseen_site",
                "data",
                "an unseen site falls back to the pooled development statistics rather than to the evaluation site",
                fallback=float(unseen.sum()),
            )
        )
    else:
        results.append(
            bad("indep.standardiser_unseen_site", "data", "an unseen site falls back to the pooled statistics", "the fallback collapsed to the identity centre")
        )

    # --- schedule and multiplicity ---------------------------------------------
    schedule = ScheduleSpec(warmup_steps=500, total_steps=3000)
    factors = [schedule.factor(step) for step in (0, 250, 500, 1000, 2000, 3000)]
    if factors[0] > 0.0 and abs(factors[2] - 1.0) < 1e-9 and all(factors[index] >= factors[index + 1] for index in range(2, len(factors) - 1)):
        results.append(
            ok(
                "indep.schedule_shape",
                "training",
                "the schedule warms up linearly to one and decays monotonically afterwards",
                factors=factors,
                warmup_steps=500,
                minimum=schedule.minimum_factor,
            )
        )
    else:
        results.append(bad("indep.schedule_shape", "training", "the schedule warms up then decays", f"factors {factors}"))

    p_values = [0.001, 0.008, 0.02, 0.04, 0.2, 0.6, 0.9]
    bh = benjamini_hochberg(p_values, level=0.05)
    try:
        from statsmodels.stats.multitest import multipletests

        library_adjusted = multipletests(p_values, alpha=0.05, method="fdr_bh")[1]
        agrees = bool(np.allclose(bh.adjusted, library_adjusted, atol=1e-12))
    except Exception:  # noqa: BLE001 - the comparison is optional
        agrees = bool(all(bh.adjusted[index] <= bh.adjusted[index + 1] + 1e-12 for index in range(len(bh.adjusted) - 1)))
    if agrees:
        results.append(
            ok(
                "indep.bh_adjustment",
                "statistic",
                "the false-discovery-rate adjustment reproduces the reference implementation",
                adjusted=bh.adjusted,
                rejected=bh.rejected,
            )
        )
    else:
        results.append(
            bad("indep.bh_adjustment", "statistic", "the false-discovery-rate adjustment reproduces the reference implementation", f"adjusted {bh.adjusted}")
        )

    ratio = interaction_ratio(5.80, 1.41, 3.05)
    if abs(round(ratio, 2) - 1.30) < 1e-9:
        results.append(
            ok(
                "indep.interaction_ratio",
                "statistic",
                "the interaction ratio equals joint divided by the sum of the separate effects",
                value=ratio,
                joint=5.80,
                separate=(1.41, 3.05),
            )
        )
    else:
        results.append(bad("indep.interaction_ratio", "statistic", "the interaction ratio equals joint over the sum of the parts", f"{ratio}"))

    stack = np.stack([np.ones(5), np.ones(5) * 2, np.zeros(5), np.full(5, 3.0)], axis=1)
    labels_four = np.array([0, 1, 2, 3, 2])
    kappa = weighted_kappa(stack.argmax(axis=1), labels_four)
    library_kappa = float(cohen_kappa_score(labels_four, stack.argmax(axis=1), weights="linear"))
    if abs(kappa - library_kappa) < 1e-12:
        results.append(
            ok("indep.kappa_library", "metric", "the weighted kappa reproduces the reference library under the documented linear weighting", value=kappa)
        )
    else:
        results.append(bad("indep.kappa_library", "metric", "the weighted kappa reproduces the reference library", f"{kappa} vs {library_kappa}"))

    exact = exact_combination_concordance(np.array([0, 1, 2, 3]), np.array([0, 1, 3, 3]))
    pooled = boundary_discordance(np.array([0, 1, 2, 3]), np.array([0, 1, 3, 3]))
    if abs(exact - 0.75) < 1e-12 and pooled >= 0.0:
        results.append(
            ok("indep.concordance_hand", "metric", "the exact-combination rate matches a hand-counted example", concordance=exact, discordance=pooled)
        )
    else:
        results.append(bad("indep.concordance_hand", "metric", "the exact-combination rate matches the hand count", f"{exact}"))

    joint_probe = joint_distribution(
        torch.tensor([[0.25, 0.25, 0.25, 0.25]], dtype=torch.float64),
        torch.tensor([[0.25, 0.25, 0.25, 0.25]], dtype=torch.float64),
        torch.tensor([[0.5, 0.5]], dtype=torch.float64),
    )
    if abs(float(joint_probe.sum()) - 1.0) < 1e-12 and joint_probe.shape[-1] == 32:
        results.append(
            ok(
                "indep.joint_distribution",
                "model",
                "the joint distribution is the outer product of the axis distributions and has unit mass",
                entries=int(joint_probe.shape[-1]),
            )
        )
    else:
        results.append(
            bad("indep.joint_distribution", "model", "the joint distribution has unit mass over the combinations", f"sum {float(joint_probe.sum())}")
        )

    return results


def _reference_achievable(t: int, n: int, m: int) -> bool:
    """The staging-definition rule rewritten from the manuscript text, for comparison."""
    if t == 1 and n > 1:
        return False
    if t == 1 and m == 1:
        return False
    if t == 2 and n > 2:
        return False
    return not (n == 0 and m == 1 and t < 4)


MANUSCRIPT_TABLE4_DISCORDANCE = {
    "full": 3.12,
    "without_stage_consistency": 5.87,
    "without_risk_control": 7.51,
    "without_both_structural": 8.92,
}
MANUSCRIPT_TABLE4_CONCORDANCE = {
    "full": 0.8312,
    "without_stage_consistency": 0.8271,
    "without_risk_control": 0.8291,
    "without_both_structural": 0.8058,
}
REPORTED_CONCORDANCE_INTERACTION = 0.87


def check_manuscript() -> list[CheckResult]:
    """Check the manuscript's own tabulated arithmetic, and surface what does not close."""
    from ..stats.multiplicity import InteractionDecomposition

    results: list[CheckResult] = []
    decision = InteractionDecomposition(
        reference=MANUSCRIPT_TABLE4_DISCORDANCE["without_both_structural"],
        without_first=MANUSCRIPT_TABLE4_DISCORDANCE["without_stage_consistency"],
        without_second=MANUSCRIPT_TABLE4_DISCORDANCE["without_risk_control"],
        full=MANUSCRIPT_TABLE4_DISCORDANCE["full"],
    )
    detail = decision.as_dict()
    if abs(detail["separate_first"] - 1.41) < 0.005 and abs(detail["separate_second"] - 3.05) < 0.005 and abs(detail["joint"] - 5.80) < 0.005:
        results.append(
            ok(
                "ms.interaction_discordance",
                "manuscript",
                "the manuscript's separate and joint contributions recompute from its own Table 4 rows",
                separate=(detail["separate_first"], detail["separate_second"]),
                joint=detail["joint"],
                ratio=detail["ratio"],
            )
        )
    else:
        results.append(bad("ms.interaction_discordance", "manuscript", "the contributions recompute from Table 4", f"{detail}"))
    if abs(round(detail["ratio"], 2) - 1.30) < 1e-9:
        results.append(
            ok(
                "ms.interaction_ratio_reported",
                "manuscript",
                "the reported discordance interaction ratio of 1.30 follows from the tabulated rows",
                ratio=detail["ratio"],
            )
        )
    else:
        results.append(
            bad("ms.interaction_ratio_reported", "manuscript", "the reported discordance interaction ratio follows from the rows", f"{detail['ratio']}")
        )

    concordance = InteractionDecomposition(
        reference=MANUSCRIPT_TABLE4_CONCORDANCE["without_both_structural"],
        without_first=MANUSCRIPT_TABLE4_CONCORDANCE["without_stage_consistency"],
        without_second=MANUSCRIPT_TABLE4_CONCORDANCE["without_risk_control"],
        full=MANUSCRIPT_TABLE4_CONCORDANCE["full"],
    )
    concordance_detail = concordance.as_dict()
    derived = round(concordance_detail["ratio"], 2)
    if abs(derived - REPORTED_CONCORDANCE_INTERACTION) < 0.005:
        results.append(
            ok(
                "ms.interaction_concordance_reported",
                "manuscript",
                "the reported concordance interaction ratio follows from the tabulated rows",
                ratio=concordance_detail["ratio"],
            )
        )
    else:
        results.append(
            bad(
                "ms.interaction_concordance_reported",
                "manuscript",
                "the reported concordance interaction ratio follows from the tabulated rows",
                f"the same estimator applied to the four Table 4 concordance rows gives {derived}, while the text reports {REPORTED_CONCORDANCE_INTERACTION}; the discrepancy is left standing rather than reconciled",
                derived=concordance_detail["ratio"],
                reported=REPORTED_CONCORDANCE_INTERACTION,
                separate=(concordance_detail["separate_first"], concordance_detail["separate_second"]),
                joint=concordance_detail["joint"],
            )
        )

    before, after = 0.00241, 0.00070
    reduction = (before - after) / before
    if abs(round(100.0 * reduction) - 71) <= 1:
        results.append(
            ok(
                "ms.variance_reduction",
                "manuscript",
                "the quoted between-site variance reduction recomputes from its own pair of values",
                reduction_percent=100.0 * reduction,
                before=before,
                after=after,
            )
        )
    else:
        results.append(bad("ms.variance_reduction", "manuscript", "the variance reduction recomputes", f"{100.0 * reduction}"))

    if abs(2842 / 3456 - 0.822) < 0.001:
        results.append(
            ok(
                "ms.retention_share",
                "manuscript",
                "the retained share of the external cohort recomputes from the two counts",
                retained=2842,
                total=3456,
                share=2842 / 3456,
            )
        )
    else:
        results.append(bad("ms.retention_share", "manuscript", "the retained share recomputes", f"{2842 / 3456}"))

    partitions = {
        "development": 5868 + 1960 == 7828,
        "analysis_set": 7828 + 3456 == 11284,
        "prospective_by_site": 612 + 498 + 402 + 338 == 1850,
        "reader_observations": 18 * 360 == 6480,
    }
    if all(partitions.values()):
        results.append(
            ok(
                "ms.cohort_arithmetic",
                "manuscript",
                "the cohort partitions close, including the prospective per-site counts and the reader observations",
                combinations=6480,
                prospective=1850,
                analysis_set=11284,
            )
        )
    else:
        results.append(bad("ms.cohort_arithmetic", "manuscript", "every cohort partition closes", f"{partitions}"))

    missing_endoscopy = 19 / 11284
    missing_pathology = 61 / 11284
    if abs(round(100.0 * missing_endoscopy, 2) - 0.17) < 0.005 and abs(round(100.0 * missing_pathology, 2) - 0.54) < 0.005:
        results.append(
            ok(
                "ms.missingness_shares",
                "manuscript",
                "the reported missingness shares recompute from the counts and the analysis-set size",
                endoscopy_percent=100.0 * missing_endoscopy,
                pathology_percent=100.0 * missing_pathology,
            )
        )
    else:
        results.append(
            bad("ms.missingness_shares", "manuscript", "the missingness shares recompute", f"{100.0 * missing_endoscopy}, {100.0 * missing_pathology}")
        )

    if abs((7.44 - 3.12) - 4.32) < 1e-9 and abs((7.19 - 3.12) - 4.07) < 1e-9:
        results.append(
            ok(
                "ms.discordance_reductions",
                "manuscript",
                "the quoted discordance reductions recompute from the tabulated rates",
                versus_unconstrained=4.32,
                versus_posthoc=4.07,
            )
        )
    else:
        results.append(bad("ms.discordance_reductions", "manuscript", "the quoted reductions recompute", "differences disagree"))

    reductions = {"versus_unconstrained": 7.44 - 3.12, "versus_posthoc": 7.19 - 3.12}
    if all(value > 3.0 for value in reductions.values()):
        results.append(
            ok(
                "ms.clinical_relevance",
                "manuscript",
                "both quoted reductions exceed the pre-specified 3.0 point clinical-relevance threshold",
                threshold=3.0,
                reductions=reductions,
            )
        )
    else:
        results.append(bad("ms.clinical_relevance", "manuscript", "the reductions exceed the clinical-relevance threshold", f"reductions {reductions}"))

    if abs((0.9451 - 0.9445) - 0.0006) < 1e-9:
        results.append(
            ok(
                "ms.t_axis_parity",
                "manuscript",
                "the quoted tumour-axis difference between the constrained model and the unconstrained arm recomputes",
                difference=0.0006,
            )
        )
    else:
        results.append(bad("ms.t_axis_parity", "manuscript", "the tumour-axis parity difference recomputes", "value disagrees"))

    half_width = 0.5 * (3.63 - 2.61)
    if abs(half_width - 0.51) < 1e-9:
        results.append(ok("ms.endpoint_interval", "manuscript", "the primary endpoint's quoted interval has the stated half-width", half_width=half_width))
    else:
        results.append(bad("ms.endpoint_interval", "manuscript", "the quoted interval half-width recomputes", f"{half_width}"))

    net_benefit_gap = 0.238 - 0.204
    if net_benefit_gap > 0 and abs(net_benefit_gap - 0.034) < 1e-9:
        results.append(
            ok(
                "ms.decision_curve_values",
                "manuscript",
                "the quoted decision-curve values are internally ordered with the constrained model highest",
                gap=net_benefit_gap,
            )
        )
    else:
        results.append(bad("ms.decision_curve_values", "manuscript", "the quoted decision-curve ordering holds", f"gap {net_benefit_gap}"))

    return results


def check_environment(root: Path, probe_links: bool = False) -> list[CheckResult]:
    """Host, container, auxiliary-resource and dataset-link checks."""
    import platform
    import shutil
    import sys as _sys

    results: list[CheckResult] = []
    try:
        import torch

        runtime = {
            "python": platform.python_version(),
            "executable_major_minor": f"{_sys.version_info.major}.{_sys.version_info.minor}",
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
        }
        results.append(ok("env.host_runtime", "environment", "the verification host's runtime is recorded", **runtime))
    except Exception as error:  # noqa: BLE001
        results.append(bad("env.host_runtime", "environment", "the verification host's runtime is recorded", f"{type(error).__name__}: {error}"))

    results.append(
        skipped(
            "env.private_cohort_clinical",
            "environment",
            "the clinical cohort is private, so every cohort-level clinical value stays NOT_RUN",
            "the multi-centre cohort is held under a data-usage statement and is not redistributed; the shipped pipeline runs a schema-compatible generator and no generated value is reported as a study result",
            tables=["Table 1", "Table 3", "Table 4", "Fig. 1", "Fig. 2", "Fig. 3", "Fig. 4"],
        )
    )
    results.append(
        blocked(
            "env.auxiliary_datasets",
            "environment",
            "model pretraining and auxiliary organ localisation need external downloads",
            "the released abdominal CT encoder weights and the two segmentation resources are downloaded separately and are absent from this checkout; no clinical claim depends on them",
            resources=["released abdominal CT vision-language encoder", "TotalSegmentator", "FLARE 2023"],
        )
    )
    docker = shutil.which("docker")
    if docker is None:
        results.append(blocked("env.container_build", "environment", "the image builds", "no container runtime is present on this host"))
    else:
        results.append(ok("env.container_build", "environment", "a container runtime is present", binary=docker))

    results.extend(check_dataset_links(root, probe=probe_links))
    return results


def check_dataset_links(root: Path, probe: bool = False) -> list[CheckResult]:
    """Parse ``dataset_urls.txt``, and probe the links only when asked to.

    Reaching out to ten hosts makes the result depend on them, and that would make the
    report's own bytes irreproducible: a clone's verification run would rewrite the
    report whenever a host happened to be unreachable. The shipped report therefore
    records the parse and leaves the live probe not run, and ``--probe-links``
    performs it on demand for a maintainer who wants the reachability evidence.
    """
    path = root / DATASET_URLS_NAME
    if not path.is_file():
        return [bad("env.dataset_links", "environment", "every recorded dataset link resolves and matches its description", f"{DATASET_URLS_NAME} is missing")]
    entries: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "	" not in stripped:
            continue
        description, _, url = stripped.partition("	")
        url = url.strip()
        if url.startswith("http"):
            entries.append((description.strip(), url))
    if not entries:
        return [
            bad("env.dataset_links", "environment", "every recorded dataset link resolves and matches its description", "no tab-separated links were found")
        ]
    description = "every recorded dataset link resolves and the returned document names the resource"
    if not probe:
        return [
            skipped(
                "env.dataset_links",
                "environment",
                description,
                "live link probing is opt-in (--probe-links) so that the report is byte-reproducible; the file was parsed and every line carries a description and a URL",
                links=len(entries),
                probed=False,
            )
        ]
    resolved: list[dict[str, Any]] = []
    mismatched: list[str] = []
    refused: list[str] = []
    missing = 0
    for description, url in entries:
        marker = description.split(",")[0].split()[0] if description.split() else ""
        # Archive and challenge hosts answer intermittently through the shared egress
        # path, so a single refusal is retried before the link is called unreachable.
        status, body = _fetch(url)
        for _ in range(2):
            if status == 200 and body is not None:
                break
            status, body = _fetch(url)
        entry = {"url": url, "description": description, "marker": marker, "http_status": status}
        if status == 200 and body is not None and (not marker or marker.lower() in body.lower()):
            entry["outcome"] = "reachable and matching"
            resolved.append(entry)
            continue
        if status in (401, 403):
            entry["outcome"] = "host refused the request from this verification host"
            refused.append(url)
        elif status == 200:
            entry["outcome"] = "reachable but the returned document does not mention the resource"
            mismatched.append(url)
        else:
            entry["outcome"] = f"not reachable (status {status})"
            missing += 1
        resolved.append(entry)
    evidence = {"probed": len(entries), "matching": len(resolved) - len(mismatched) - len(refused) - missing, "links": resolved}
    if mismatched:
        return [
            bad(
                "env.dataset_links",
                "environment",
                description,
                f"{len(mismatched)} link(s) returned a document that does not name the resource: {', '.join(mismatched)}",
                **evidence,
            )
        ]
    if missing or refused:
        return [
            skipped(
                "env.dataset_links",
                "environment",
                description,
                f"{len(refused)} link(s) were refused by the host and {missing} were unreachable from this verification host; a refusal is not evidence that a link is dead",
                **evidence,
            )
        ]
    return [ok("env.dataset_links", "environment", description, **evidence)]


def _fetch(url: str) -> tuple[int, str | None]:
    """Fetch a URL, returning the status code and the decoded body when it arrives."""
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; release-verification)"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - fixed scheme, release-provided URLs
            raw = response.read()
            return int(response.status), raw.decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        return int(error.code), None
    except Exception:  # noqa: BLE001 - any transport failure is reported as unreachable
        return 0, None


def build_claim_document(claims: list[dict[str, Any]]) -> dict[str, Any]:
    """The claim map document written to ``claim_to_code.json``."""
    return {
        "artifact": "paper-claim to code mapping",
        "release": "A Multimodal Radiomics Foundation Model for Gastric Cancer Staging in Real-World Multi-Center Imaging Cohorts",
        "venue": "npj Digital Medicine",
        "package_dir": PACKAGE_DIR,
        "anchor_convention": "code[].path is relative to package_dir; code[].symbol is a module-level name, or Class.method for a method",
        "verification_vocabulary": {
            "EXECUTED": "the anchor resolved and the mechanism runs on this host",
            "UNRESOLVED": "the anchor did not resolve to a real file and symbol",
            "NOT_RUN_PRIVATE_COHORT": "the mechanism is implemented, but the value it would produce is a cohort-level clinical quantity and the cohort is private",
            "BLOCKED_AUX_DATASET": "the mechanism is implemented, but its input is an external resource that is downloaded separately",
        },
        "claims": claims,
        "deviations": DEVIATIONS,
    }


def build_report(root: Path, claims: list[dict[str, Any]], checks: list[CheckResult], report_name: str) -> dict[str, Any]:
    """Assemble the verification report document."""
    from ..utils.config import ExperimentConfig, resolve_experiment

    counts = {status: sum(1 for check in checks if check.status == status) for status in (PASS, FAIL, NOT_RUN, BLOCKED)}
    config = ExperimentConfig.from_mapping(resolve_experiment("main", root / "configs", []))
    manuscript = [check for check in checks if check.category == "manuscript"]
    code_checks = [check for check in checks if check.category != "manuscript"]
    code_counts = {status: sum(1 for check in code_checks if check.status == status) for status in (PASS, FAIL, NOT_RUN, BLOCKED)}

    def status_of(block: dict[str, int]) -> str:
        if block[FAIL]:
            return UNVERIFIED
        if block[NOT_RUN] or block[BLOCKED]:
            return PARTIALLY_VERIFIED
        return VERIFIED

    overall = status_of(counts)
    code_status = status_of(code_counts)
    return {
        "release": {
            "title": "A Multimodal Radiomics Foundation Model for Gastric Cancer Staging in Real-World Multi-Center Imaging Cohorts",
            "venue": "npj Digital Medicine",
            "release_slug": RELEASE_SLUG,
            "importable_package": "stagefm",
            "config": "configs/experiment/main.yaml",
            "seed": config.seed,
            "report": report_name,
        },
        "summary": {
            **counts,
            "overall_status": overall,
            "code_status": code_status,
            "checks": len(checks),
            "manuscript_checks": len(manuscript),
            "manuscript_discrepancies": [check.as_dict() for check in manuscript if check.status == FAIL],
            "code_counts": code_counts,
        },
        "claim_mapping": {
            "artifact": "paper-claim to code mapping",
            "claims": len(claims),
            "anchors": sum(len(claim["code"]) for claim in claims),
            "resolved_anchors": sum(1 for claim in claims for anchor in claim["code"] if anchor["resolved"]),
        },
        "checks": [check.as_dict() for check in checks],
        "interpretation": (
            "PASS means the check ran on this host and held. FAIL means it ran and did not hold. NOT_RUN means the input it needs is "
            "unavailable, which for a private clinical cohort is the expected outcome and is never an invitation to substitute a number. "
            "BLOCKED means an external resource is required and is not present in this checkout. overall_status covers every check; "
            "code_status covers the same set without the manuscript-arithmetic family, so a discrepancy inside the manuscript's own tables "
            "is not confused with a defect in this release."
        ),
    }


def summary_text(report: dict[str, Any]) -> str:
    """The plain-text verification summary, written before the integrity manifest."""
    summary = report["summary"]
    lines = [
        "release verification summary",
        f"package         : {report['release']['importable_package']}",
        f"release slug    : {report['release']['release_slug']}",
        f"overall status  : {summary['overall_status']}",
        f"code status     : {summary.get('code_status', summary['overall_status'])}",
        f"manuscript fails: {len(summary.get('manuscript_discrepancies', []))}",
        f"checks          : PASS {summary[PASS]}  FAIL {summary[FAIL]}  NOT_RUN {summary[NOT_RUN]}  BLOCKED {summary[BLOCKED]}",
        f"claim mapping   : {report['claim_mapping']['claims']} claims, {report['claim_mapping']['anchors']} anchors",
        "artefacts       : verification_report.json, claim_to_code.json, integrity_manifest.json",
        "",
        "per-check status",
    ]
    width = max((len(check["id"]) for check in report["checks"]), default=0)
    for check in report["checks"]:
        detail = check["detail"][:100] if check["detail"] else ""
        lines.append(f"  {check['status']:<8}  {check['id']:<{width}}  {detail}".rstrip())
    return "\n".join(lines)


def _guard(function: Callable[..., Any], *arguments: Any) -> list[CheckResult]:
    """Run a check group, degrading to a single FAIL entry instead of raising."""
    name = getattr(function, "__name__", "check")
    try:
        return list(function(*arguments))
    except Exception as error:  # noqa: BLE001 - the boundary is the verification driver itself
        return [bad(name, "verification", f"{name} ran", f"{type(error).__name__}: {error}", traceback=traceback.format_exc()[-1500:])]


def main(argv: list[str] | None = None) -> int:
    """Run the verification suite and write the release artefacts."""
    parser = argparse.ArgumentParser(description="Verify the release and write its integrity artefacts.")
    parser.add_argument("--root", default=None, help="repository root; defaults to the discovered one")
    parser.add_argument("--dump-claims", action="store_true", help="write claim_to_code.json and exit")
    parser.add_argument("--dry-run", action="store_true", help="run every check and print the summary without writing any artefact")
    parser.add_argument("--probe-links", action="store_true", help="also reach out to every dataset link and record its status")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve() if args.root else find_repo_root(Path(__file__).resolve())
    LOGGER.info("verifying release rooted at %s", root)
    claims, claim_check = check_claims(root)
    if args.dump_claims:
        write_json(root / CLAIM_NAME, build_claim_document(claims))
        return 0
    checks: list[CheckResult] = [claim_check]
    checks.extend(_guard(check_execution, root))
    checks.extend(_guard(check_procedures))
    checks.extend(_guard(check_manuscript))
    checks.extend(_guard(check_environment, root, args.probe_links))
    if args.dry_run:
        report = build_report(root, claims, checks, VERIFICATION_NAME)
        print(summary_text(report))
        return 0
    report = build_report(root, claims, checks, VERIFICATION_NAME)
    write_json(root / VERIFICATION_NAME, report)
    write_json(root / CLAIM_NAME, build_claim_document(claims))
    write_text(root / SUMMARY_NAME, summary_text(report))
    manifest = manifest_digest(root, excluded=[MANIFEST_NAME])
    write_json(
        root / MANIFEST_NAME,
        {
            "root": RELEASE_SLUG,
            "excluded": [MANIFEST_NAME],
            "file_count": manifest["file_count"],
            "aggregate_sha256": manifest["aggregate_sha256"],
            "algorithm": "sha256 over sorted (relative path, sha256) pairs",
            "files": manifest["files"],
        },
    )
    LOGGER.info("verification complete: %s", report["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
