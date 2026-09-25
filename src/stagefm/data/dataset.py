"""Torch datasets and the batch contract used by training and evaluation.

Each item carries the raw examination volume plus the four non-imaging streams and
the supervision. Volumes are produced by the schema-compatible generator because
the clinical volumes are not redistributable; the generator's latent is stored per
record, so a volume is reproducible from the record alone.

The batch is a plain dictionary rather than a dataclass because PyTorch's default
collate handles a homogeneous tensor dictionary directly and the trainer reads it
once per step.

Ref: Methods Sec. 4.3 (preprocessing), Sec. 4.5 (streams), Sec. 4.7 (batch size,
fixed patch grid).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ..utils.config import ExperimentConfig, ImagingConfig
from .cohort import Layer
from .imaging import augment_volume, synthesise_patient_volume
from .schema import Examination
from .streams import StreamPlan, apply_modality_dropout, dropout_rates, presence_vector
from .text import TextVocabulary

MAX_TERMS = 6


@dataclass(frozen=True)
class DatasetSpec:
    """Everything an :class:`ExaminationDataset` needs beyond the records."""

    imaging: ImagingConfig
    vocabulary: TextVocabulary
    plan: StreamPlan
    patch_grid: tuple[int, int, int]
    modality_dropout: float
    training: bool
    seed: int
    site_index: dict[str, int]


class ExaminationDataset(Dataset[dict[str, Any]]):
    """One examination per item, with on-demand volume synthesis."""

    def __init__(self, records: list[Examination], spec: DatasetSpec) -> None:
        self._records = records
        self._spec = spec

    def __len__(self) -> int:
        return len(self._records)

    @property
    def records(self) -> list[Examination]:
        return self._records

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self._records[index]
        spec = self._spec
        rng = np.random.default_rng(spec.seed + index)
        latent = record.imaging_latent if record.imaging_latent is not None else np.zeros(32, dtype=np.float32)
        site_position = spec.site_index.get(record.site, 0)
        volume = synthesise_patient_volume(latent, spec.patch_grid, spec.imaging, site_position, spec.seed + 31 * index)
        if spec.training:
            volume = augment_volume(
                volume,
                rng,
                spec.imaging.rotate_degrees,
                spec.imaging.scale_fraction,
                spec.imaging.intensity_jitter_hu,
            )

        availability = dict(record.stream_availability())
        if spec.training:
            availability = apply_modality_dropout(availability, dropout_rates(spec.modality_dropout), rng, spec.plan)

        radiomics = record.radiomics if record.radiomics is not None else np.zeros(1, dtype=np.float32)
        clinical = record.clinical if record.clinical is not None else np.zeros(1, dtype=np.float32)
        endoscopy = spec.vocabulary.encode_endoscopy(record.endoscopy_terms if availability.get("endoscopy", False) else ())
        pathology = spec.vocabulary.encode_pathology(record.pathology_terms if availability.get("pathology", False) else ())

        item: dict[str, Any] = {
            "record_id": record.record_id,
            "site": record.site,
            "volume": torch.from_numpy(np.ascontiguousarray(volume)[None, ...]),
            "radiomics": torch.from_numpy(np.asarray(radiomics, dtype=np.float32)),
            "clinical": torch.from_numpy(np.asarray(clinical, dtype=np.float32)),
            "endoscopy": _pad_terms(endoscopy),
            "pathology": _pad_terms(pathology),
            "presence": torch.from_numpy(presence_vector(availability, spec.plan)),
            "label_t": torch.tensor(record.stage.t - 1, dtype=torch.long),
            "label_n": torch.tensor(record.stage.n, dtype=torch.long),
            "label_m": torch.tensor(record.stage.m, dtype=torch.long),
            "stage_column": torch.tensor(record.stage.flat_index, dtype=torch.long),
            "harvested_nodes": torch.tensor(record.harvested_nodes, dtype=torch.long),
            "site_index": torch.tensor(site_position, dtype=torch.long),
        }
        return item


def _pad_terms(indices: tuple[int, ...]) -> torch.Tensor:
    """Left-pad a term list to a fixed length with the absence index."""
    trimmed = indices[:MAX_TERMS]
    padded = (0,) * (MAX_TERMS - len(trimmed)) + trimmed
    return torch.tensor(padded, dtype=torch.long)


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack the tensor fields and collect the identifiers as lists."""
    tensor_keys = [key for key in batch[0] if isinstance(batch[0][key], torch.Tensor)]
    out: dict[str, Any] = {key: torch.stack([item[key] for item in batch], dim=0) for key in tensor_keys}
    out["record_id"] = [item["record_id"] for item in batch]
    out["site"] = [item["site"] for item in batch]
    return out


def build_datasets(
    config: ExperimentConfig,
    layers: dict[Layer, list[Examination]],
    vocabulary: TextVocabulary,
    plan: StreamPlan,
    training_shuffle_seed: int = 0,
) -> dict[Layer, ExaminationDataset]:
    """Wrap each cohort layer in a dataset with the arm's stream plan."""
    sites = sorted({record.site for record in layers[Layer.TRAIN] + layers[Layer.EXTERNAL]})
    site_index = {site: position for position, site in enumerate(sites)}
    datasets: dict[Layer, ExaminationDataset] = {}
    for position, layer in enumerate((Layer.TRAIN, Layer.INTERNAL_TEST, Layer.EXTERNAL, Layer.PROSPECTIVE)):
        records = list(layers[layer])
        spec = DatasetSpec(
            imaging=config.imaging,
            vocabulary=vocabulary,
            plan=plan,
            patch_grid=config.imaging.patch_grid,
            modality_dropout=config.fusion.modality_dropout,
            training=layer is Layer.TRAIN,
            seed=training_shuffle_seed + 7919 * position,
            site_index=site_index,
        )
        datasets[layer] = ExaminationDataset(records, spec)
    return datasets


def label_table(dataset: ExaminationDataset) -> dict[str, np.ndarray]:
    """Integer label arrays for a dataset, used by the metrics stage."""
    records = dataset.records
    return {
        "t": np.array([record.stage.t - 1 for record in records], dtype=np.int64),
        "n": np.array([record.stage.n for record in records], dtype=np.int64),
        "m": np.array([record.stage.m for record in records], dtype=np.int64),
        "stage_column": np.array([record.stage.flat_index for record in records], dtype=np.int64),
        "harvested_nodes": np.array([record.harvested_nodes for record in records], dtype=np.int64),
        "site": np.array([record.site for record in records]),
    }
