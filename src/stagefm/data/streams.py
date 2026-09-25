"""Stream availability, modality dropout and the absence token.

The fusion block consumes the encoder's patch tokens plus the non-imaging streams.
Streams differ in dimension and in missingness pattern, and two of them -- the
endoscopic description and the pathology text -- are frequently unavailable at the
point of care. Training therefore applies modality dropout independently per
stream so that the model remains operational when a stream is absent, and an absent
stream is signalled by a learned absence token rather than imputed.

The ablation arms are expressed as a :class:`StreamPlan`, so removing a stream is a
configuration change rather than a code path.

Ref: Methods Sec. 4.5 (fusion block, modality dropout, absence token); Sec. 4.6
(the 0.15 dropout threshold); Sec. 4.7 (augmentations).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils.config import ExperimentConfig
from .schema import Stream

STREAM_ORDER: tuple[Stream, ...] = (Stream.CT, Stream.RADIOMICS, Stream.CLINICAL, Stream.ENDOSCOPY, Stream.PATHOLOGY)
NON_IMAGING_ORDER: tuple[Stream, ...] = (Stream.RADIOMICS, Stream.CLINICAL, Stream.ENDOSCOPY, Stream.PATHOLOGY)
TEXT_STREAMS: tuple[Stream, ...] = (Stream.ENDOSCOPY, Stream.PATHOLOGY)


@dataclass(frozen=True)
class StreamPlan:
    """Which streams an arm feeds into the fusion block."""

    active: tuple[Stream, ...]

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> StreamPlan:
        active: list[Stream] = []
        for stream in STREAM_ORDER:
            if stream is Stream.RADIOMICS:
                if config.ablation_enabled("radiomics"):
                    active.append(stream)
                continue
            if config.ablation_enabled(stream.value):
                active.append(stream)
        return cls(active=tuple(active))

    def includes(self, stream: Stream) -> bool:
        return stream in self.active

    @property
    def non_imaging(self) -> tuple[Stream, ...]:
        return tuple(stream for stream in self.active if stream is not Stream.CT)

    @property
    def drops_imaging(self) -> bool:
        return Stream.CT not in self.active

    def drop_imaging(self) -> StreamPlan:
        """The arm in which the image stream is removed."""
        return StreamPlan(active=tuple(stream for stream in self.active if stream is not Stream.CT))


def dropout_rates(modality_dropout: float, overrides: dict[str, float] | None = None) -> dict[Stream, float]:
    """Per-stream dropout probability.

    The manuscript names clinical, endoscopic and pathology at 0.15. The radiomic
    block is a fourth non-imaging stream with the same missingness behaviour, so it
    carries the same rate unless a config override says otherwise.
    """
    rates = {stream: float(modality_dropout) for stream in NON_IMAGING_ORDER}
    for key, value in (overrides or {}).items():
        rates[Stream(key)] = float(value)
    return rates


def apply_modality_dropout(
    availability: dict[str, bool],
    rates: dict[Stream, float],
    rng: np.random.Generator,
    plan: StreamPlan,
) -> dict[str, bool]:
    """Independently blank available non-imaging streams at their configured rate.

    A stream that is already unavailable stays unavailable; the operation only ever
    removes information, never invents it.
    """
    out = dict(availability)
    for stream in plan.non_imaging:
        if not out.get(stream.value, False):
            continue
        rate = rates.get(stream, 0.0)
        if rate > 0.0 and rng.random() < rate:
            out[stream.value] = False
    return out


def presence_vector(availability: dict[str, bool], plan: StreamPlan) -> np.ndarray:
    """One indicator per active stream, in :data:`STREAM_ORDER`."""
    return np.array([1.0 if availability.get(stream.value, False) else 0.0 for stream in plan.active], dtype=np.float32)


def stream_channels(plan: StreamPlan) -> int:
    """Number of streams the fusion block will attend over (including the image stream)."""
    return len(plan.active)


def missing_stream_counts(records: list[object], attribute: str) -> dict[str, int | str]:
    """How many records lack a given stream, for the missingness report."""
    total = 0
    for record in records:
        availability = record.stream_availability()  # type: ignore[attr-defined]
        if not availability.get(attribute, False):
            total += 1
    return {"stream": attribute, "missing": total, "total": len(records)}
