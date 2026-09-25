"""Calibration loss.

The calibration error the manuscript reports is measured per site, so the training
objective penalises the site-level gap between the mean predicted probability and
the observed frequency rather than a global confidence gap. Two variants are
available: the systematic gap, which has the site-conditional correction as its
natural target, and a Brier term over the systemic mass.

Ref: Algorithm 1 step 9 (third term); Methods Sec. 4.5 (site-conditional
calibration); Sec. 4.11 (per-site expected calibration error).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .decision import systemic_mass


@dataclass(frozen=True)
class CalibrationLoss:
    """Total calibration penalty and the per-site gaps that produced it."""

    total: Tensor
    per_site_gap: dict[int, Tensor]


def site_gap_loss(probability: Tensor, targets: Tensor, site_index: Tensor, min_count: int = 1) -> CalibrationLoss:
    """Mean absolute site-level difference between mean probability and mean outcome."""
    gaps: dict[int, Tensor] = {}
    total = probability.sum() * 0.0
    sites = torch.unique(site_index)
    for site in sites.tolist():
        selector = site_index == site
        count = int(selector.sum().item())
        if count < min_count:
            continue
        gap = (probability[selector].mean() - targets[selector].mean()).abs()
        gaps[int(site)] = gap
        total = total + gap
    if gaps:
        total = total / len(gaps)
    return CalibrationLoss(total=total, per_site_gap=gaps)


def brier_surrogate(probability: Tensor, targets: Tensor) -> Tensor:
    """Brier score of the systemic mass."""
    return F.mse_loss(probability, targets)


def calibration_loss(projected: Tensor, reference_stages: Tensor, site_index: Tensor, variant: str = "site_gap") -> CalibrationLoss:
    """Site-conditional calibration penalty of one batch."""
    from ..data.schema import ALL_TRIPLES  # local import keeps the module free of a cycle at import time
    from ..data.staging import treatment_category

    order = {"surgery_first": 0, "perioperative": 1, "systemic": 2}
    mapping = torch.tensor(
        [order[treatment_category(stage).value] for stage in ALL_TRIPLES[: projected.shape[-1]]],
        dtype=torch.long,
        device=projected.device,
    )
    targets = (mapping[reference_stages] == 2).to(projected.dtype)
    probability = systemic_mass(projected)
    if variant == "brier":
        return CalibrationLoss(total=brier_surrogate(probability, targets), per_site_gap={})
    return site_gap_loss(probability, targets, site_index)
