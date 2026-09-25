"""Inference loop and the prediction bundle every downstream analysis reads.

One forward pass over a loader is collected into a single bundle, and every later
analysis -- discrimination, calibration, the decision endpoint, per-site and
subgroup breakdowns, the ascertainment decomposition -- reads from that bundle
rather than re-running the model. That keeps the reported numbers consistent with
each other and makes the verification pass able to recompute them from stored
predictions.

Ref: Methods Sec. 4.11 (evaluation protocol); Algorithm 1 (inference).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..training.amp import PrecisionSpec, autocast_context, to_device
from ..utils.logging import get_logger

LOGGER = get_logger("evaluation.loop")


@dataclass
class PredictionBundle:
    """Predictions and supervision for one evaluation layer."""

    t_prob: np.ndarray
    n_prob: np.ndarray
    m_prob: np.ndarray
    joint: np.ndarray
    projected: np.ndarray
    boundary_logit: np.ndarray
    labels: dict[str, np.ndarray]
    sites: list[str]
    harvested_nodes: np.ndarray
    record_ids: list[str]
    stage_column: np.ndarray
    extra: dict[str, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.stage_column.shape[0])

    @property
    def predicted_columns(self) -> np.ndarray:
        """Argmax decode of the projected distribution."""
        return np.asarray(self.projected.argmax(axis=-1), dtype=np.int64)

    def rows(self) -> list[dict[str, Any]]:
        """One dictionary per examination, for the case tables."""
        return [
            {
                "record_id": self.record_ids[index],
                "site": self.sites[index],
                "harvested_nodes": int(self.harvested_nodes[index]),
                "reference_column": int(self.stage_column[index]),
                "predicted_column": int(self.predicted_columns[index]),
                "boundary_logit": float(self.boundary_logit[index]),
            }
            for index in range(len(self))
        ]

    def subset(self, selector: np.ndarray) -> PredictionBundle:
        """Restrict to a boolean mask over examinations."""
        flags = np.asarray(selector, dtype=bool)
        if flags.shape != (len(self),):
            raise ValueError(f"selector must be a length-{len(self)} boolean mask")
        return PredictionBundle(
            t_prob=self.t_prob[flags],
            n_prob=self.n_prob[flags],
            m_prob=self.m_prob[flags],
            joint=self.joint[flags],
            projected=self.projected[flags],
            boundary_logit=self.boundary_logit[flags],
            labels={key: value[flags] for key, value in self.labels.items()},
            sites=[site for site, keep in zip(self.sites, flags.tolist()) if keep],
            harvested_nodes=self.harvested_nodes[flags],
            record_ids=[name for name, keep in zip(self.record_ids, flags.tolist()) if keep],
            stage_column=self.stage_column[flags],
            extra={key: value[flags] for key, value in self.extra.items()},
        )

    def as_state(self) -> dict[str, object]:
        """Serialisable view, used by the verification pass."""
        return {
            "t_prob": self.t_prob,
            "n_prob": self.n_prob,
            "m_prob": self.m_prob,
            "joint": self.joint,
            "projected": self.projected,
            "boundary_logit": self.boundary_logit,
            "stage_column": self.stage_column,
            "sites": self.sites,
            "harvested_nodes": self.harvested_nodes,
        }


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    spec: PrecisionSpec,
    device: torch.device,
) -> PredictionBundle:
    """Run the model over a loader and collect every quantity the analyses need."""
    model.eval()
    t_prob: list[np.ndarray] = []
    n_prob: list[np.ndarray] = []
    m_prob: list[np.ndarray] = []
    joint: list[np.ndarray] = []
    projected: list[np.ndarray] = []
    boundary: list[np.ndarray] = []
    stages: list[np.ndarray] = []
    labels: dict[str, list[np.ndarray]] = {"t": [], "n": [], "m": []}
    sites: list[str] = []
    nodes: list[np.ndarray] = []
    identifiers: list[str] = []
    for raw_batch in loader:
        batch = to_device(raw_batch, device)
        with autocast_context(spec):
            output = model(batch)
        t_prob.append(_numpy(output.axis["T"].probabilities))
        n_prob.append(_numpy(output.axis["N"].probabilities))
        m_prob.append(_numpy(output.axis["M"].probabilities))
        joint.append(_numpy(output.joint))
        projected.append(_numpy(output.projected))
        boundary.append(_numpy(output.boundary_logit))
        stages.append(_numpy(batch["stage_column"]).astype(np.int64))
        for key in ("t", "n", "m"):
            labels[key].append(_numpy(batch[f"label_{key}"]).astype(np.int64))
        sites.extend(cast(list[str], batch["site"]))
        nodes.append(_numpy(batch["harvested_nodes"]).astype(np.int64))
        identifiers.extend(cast(list[str], batch["record_id"]))
    bundle = PredictionBundle(
        t_prob=np.concatenate(t_prob, axis=0),
        n_prob=np.concatenate(n_prob, axis=0),
        m_prob=np.concatenate(m_prob, axis=0),
        joint=np.concatenate(joint, axis=0),
        projected=np.concatenate(projected, axis=0),
        boundary_logit=np.concatenate(boundary, axis=0),
        labels={key: np.concatenate(value, axis=0) for key, value in labels.items()},
        sites=sites,
        harvested_nodes=np.concatenate(nodes, axis=0),
        record_ids=identifiers,
        stage_column=np.concatenate(stages, axis=0),
    )
    LOGGER.info("collected %d predictions", len(bundle))
    return bundle


def _numpy(value: Any) -> np.ndarray:
    tensor = value.detach() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    return tensor.to(torch.float32).cpu().numpy()


def category_mass_table(bundle: PredictionBundle) -> np.ndarray:
    """Per-record mass of each management category under the projected distribution."""
    from ..models.risk_control import category_masses

    return category_masses(bundle.projected)


def boundary_truth(bundle: PredictionBundle) -> np.ndarray:
    """Reference management category index of every record."""
    from ..data.schema import ALL_TRIPLES
    from ..models.risk_control import reference_categories

    return reference_categories([ALL_TRIPLES[int(column)] for column in bundle.stage_column])
