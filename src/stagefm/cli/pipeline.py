"""Pipeline assembly shared by the training and evaluation entry points.

The cohort, the term vocabularies, the stream plan, the arm and the data loaders are
built here so that training and evaluation cannot disagree about any of them: an
evaluation run rebuilds the same objects from the same configuration and the same
seed, which is what makes a stored results table reproducible.

Ref: Methods Sec. 4.1 (layers), Sec. 4.4 (public auxiliary resources), Sec. 4.5
(streams), Sec. 4.7 (seeds and tuning budget).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ..data.cohort import Layer, build_site_profiles
from ..data.dataset import ExaminationDataset, build_datasets, collate
from ..data.schema import Examination
from ..data.staging import AchievableRule, AchievableSet
from ..data.streams import StreamPlan
from ..data.synthetic import SyntheticCohortConfig, generate_cohort
from ..data.text import TextVocabulary
from ..models.baselines import StreamDimensions, build_arm
from ..training.trainer import loader_for
from ..utils.config import ExperimentConfig
from ..utils.logging import get_logger
from ..utils.seeding import set_seed

LOGGER = get_logger("cli.pipeline")


@dataclass
class PreparedPipeline:
    """Everything an entry point needs to train or evaluate one arm."""

    config: ExperimentConfig
    layers: dict[Layer, list[Examination]]
    datasets: dict[Layer, ExaminationDataset]
    vocabulary: TextVocabulary
    plan: StreamPlan
    achievable: AchievableSet
    model: torch.nn.Module
    dimensions: StreamDimensions
    site_index: dict[str, int]

    def loader(self, layer: Layer, shuffle: bool = False) -> DataLoader[dict[str, Any]]:
        """A loader for one cohort layer."""
        return loader_for(
            self.datasets[layer],
            batch_size=self.config.train.batch_size,
            shuffle=shuffle,
            num_workers=self.config.train.num_workers,
            seed=self.config.seed,
            collate_fn=collate,
        )

    def records(self, layer: Layer) -> list[Examination]:
        return self.layers[layer]


def achievable_for(config: ExperimentConfig) -> AchievableSet:
    """The achievable set named by the arm's configuration."""
    rule = str(config.evaluation.get("achievable_rule", AchievableRule.STAGING.value))
    return AchievableSet.from_rule(rule)


def prepare(config: ExperimentConfig, arm: str = "stagefm", seed: int | None = None) -> PreparedPipeline:
    """Build the cohort, the datasets and the arm from a resolved configuration."""
    chosen_seed = config.seed if seed is None else seed
    set_seed(chosen_seed)
    achievable = achievable_for(config)
    synthetic = SyntheticCohortConfig.from_mapping(dict(config.evaluation.get("synthetic", {})))
    train, internal, external, prospective, profiles = generate_cohort(config.cohort, achievable, synthetic, seed=chosen_seed)
    layers: dict[Layer, list[Examination]] = {
        Layer.TRAIN: train,
        Layer.INTERNAL_TEST: internal,
        Layer.EXTERNAL: external,
        Layer.PROSPECTIVE: prospective,
    }
    vocabulary = TextVocabulary.fit(train)
    plan = StreamPlan.from_config(config)
    datasets = build_datasets(config, layers, vocabulary, plan, training_shuffle_seed=chosen_seed)
    dimensions = StreamDimensions(
        radiomics=int(next(iter(datasets[Layer.TRAIN].records)).radiomics.shape[0]),  # type: ignore[union-attr]
        clinical=int(next(iter(datasets[Layer.TRAIN].records)).clinical.shape[0]),  # type: ignore[union-attr]
        endoscopy_vocab=vocabulary.endoscopy_size,
        pathology_vocab=vocabulary.pathology_size,
    )
    model = build_arm(arm, config, plan, achievable, dimensions)
    site_index = {site: position for position, site in enumerate(sorted(profiles))}
    LOGGER.info(
        "prepared arm=%s layers=%s active_streams=%s",
        arm,
        {layer.value: len(records) for layer, records in layers.items()},
        [stream.value for stream in plan.active],
    )
    return PreparedPipeline(
        config=config,
        layers=layers,
        datasets=datasets,
        vocabulary=vocabulary,
        plan=plan,
        achievable=achievable,
        model=model,
        dimensions=dimensions,
        site_index=site_index,
    )


def output_root(config: ExperimentConfig, arm: str) -> Path:
    """Directory a run writes into."""
    base = Path(str(config.evaluation.get("output_dir", "runs")))
    return base / config.name / arm


def profile_table(config: ExperimentConfig) -> list[dict[str, Any]]:
    """The site table as a list of dictionaries, for the run metadata."""
    profiles = build_site_profiles(config.cohort)
    return [
        {
            "site": profile.site,
            "region": profile.region,
            "median_harvested_nodes": profile.median_harvested_nodes,
            "scanner_vendor": profile.scanner_vendor,
            "development": profile.development,
        }
        for profile in profiles.values()
    ]
