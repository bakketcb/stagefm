"""Cohort contract, imaging preprocessing, radiomic descriptors and input streams.

The package is organised so that the pieces the manuscript fixes before training --
the achievable label set, the treatment-boundary rule, the shell width, the
stability and redundancy filters, the per-site standardisation -- are all
constructible without touching the network code.
"""

from __future__ import annotations

from .cohort import Layer, NodeStratum, SiteProfile, build_site_profiles, site_case_shares, stratum_of, subsample_to_cutoff
from .dataset import DatasetSpec, ExaminationDataset, build_datasets, collate, label_table
from .imaging import augment_volume, patchify, preprocess_volume, synthesise_volume, token_grid
from .radiomics import DescriptorSet, SiteStandardiser, collapse_redundant, extract_descriptors, stability_select
from .schema import ALL_TRIPLES, NON_IMAGING_STREAMS, Axis, Examination, StageTriple, Stream, TreatmentCategory
from .segmentation import OrganMasks, localise, peritumoral_shell, stability_perturbations
from .staging import AchievableRule, AchievableSet, boundary_discordant, shift_category, treatment_category
from .streams import NON_IMAGING_ORDER, STREAM_ORDER, StreamPlan, apply_modality_dropout, dropout_rates, presence_vector
from .synthetic import SyntheticCohortConfig, generate_cohort, strata_report
from .text import ABSENT_TERM, TextVocabulary, as_text_inputs

__all__ = [
    "ABSENT_TERM",
    "ALL_TRIPLES",
    "NON_IMAGING_ORDER",
    "NON_IMAGING_STREAMS",
    "STREAM_ORDER",
    "AchievableRule",
    "AchievableSet",
    "Axis",
    "DatasetSpec",
    "DescriptorSet",
    "Examination",
    "ExaminationDataset",
    "Layer",
    "NodeStratum",
    "OrganMasks",
    "SiteProfile",
    "SiteStandardiser",
    "StageTriple",
    "Stream",
    "StreamPlan",
    "SyntheticCohortConfig",
    "TextVocabulary",
    "TreatmentCategory",
    "apply_modality_dropout",
    "as_text_inputs",
    "augment_volume",
    "boundary_discordant",
    "build_datasets",
    "build_site_profiles",
    "collate",
    "collapse_redundant",
    "dropout_rates",
    "extract_descriptors",
    "generate_cohort",
    "label_table",
    "localise",
    "patchify",
    "peritumoral_shell",
    "presence_vector",
    "preprocess_volume",
    "shift_category",
    "site_case_shares",
    "stability_perturbations",
    "stability_select",
    "strata_report",
    "stratum_of",
    "subsample_to_cutoff",
    "synthesise_volume",
    "token_grid",
    "treatment_category",
]
