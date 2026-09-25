"""Cohort schema: stage labels, treatment categories and one examination record.

The label axes follow the manuscript's ground-truth definition (Methods Sec. 4.1):
the T axis is a four-category grouping T1-T4 with the T4a/T4b subdivisions merged,
the N axis is ordinal over N0-N3, and the M axis is binary. The treatment category
is derived from the stage by the boundary rule in :mod:`stagefm.data.staging`, so a
site identifier never enters the label.

Ref: Methods Sec. 4.1 (ground truth and treatment-boundary labels).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

T_CLASSES = 4
N_CLASSES = 4
M_CLASSES = 2


class Axis(str, Enum):
    """The three staging axes."""

    T = "T"
    N = "N"
    M = "M"


class TreatmentCategory(str, Enum):
    """Management categories the boundary rule maps a stage onto."""

    SURGERY_FIRST = "surgery_first"
    PERIOPERATIVE = "perioperative"
    SYSTEMIC = "systemic"


class Stream(str, Enum):
    """The five input streams the fusion block consumes."""

    CT = "ct"
    RADIOMICS = "radiomics"
    CLINICAL = "clinical"
    ENDOSCOPY = "endoscopy"
    PATHOLOGY = "pathology"


NON_IMAGING_STREAMS: tuple[Stream, ...] = (Stream.RADIOMICS, Stream.CLINICAL, Stream.ENDOSCOPY, Stream.PATHOLOGY)


@dataclass(frozen=True, order=True)
class StageTriple:
    """One joint stage label, indexed from one on T and zero on N and M."""

    t: int
    n: int
    m: int

    def __post_init__(self) -> None:
        if not 1 <= self.t <= T_CLASSES:
            raise ValueError(f"T category out of range: {self.t}")
        if not 0 <= self.n < N_CLASSES:
            raise ValueError(f"N category out of range: {self.n}")
        if not 0 <= self.m < M_CLASSES:
            raise ValueError(f"M category out of range: {self.m}")

    @property
    def flat_index(self) -> int:
        """Row-major index into a length 32 joint distribution."""
        return (self.t - 1) * (N_CLASSES * M_CLASSES) + self.n * M_CLASSES + self.m

    @classmethod
    def from_flat_index(cls, index: int) -> StageTriple:
        t = index // (N_CLASSES * M_CLASSES) + 1
        remainder = index % (N_CLASSES * M_CLASSES)
        return cls(t=t, n=remainder // M_CLASSES, m=remainder % M_CLASSES)

    def label(self) -> str:
        """The clinical shorthand, for logs and case tables only."""
        return f"T{self.t}N{self.n}M{self.m}"


ALL_TRIPLES: tuple[StageTriple, ...] = tuple(StageTriple.from_flat_index(index) for index in range(T_CLASSES * N_CLASSES * M_CLASSES))
_STAGE_TO_COLUMN = {stage.flat_index: column for column, stage in enumerate(ALL_TRIPLES)}


def label_matrix() -> np.ndarray:
    """``(32, 3)`` integer matrix of every joint label, ordered by flat index."""
    return np.array([[stage.t, stage.n, stage.m] for stage in ALL_TRIPLES], dtype=np.int64)


def column_of(stage: StageTriple) -> int:
    """Position of ``stage`` in :data:`ALL_TRIPLES`."""
    return _STAGE_TO_COLUMN[stage.flat_index]


@dataclass
class Examination:
    """One de-identified examination and its supervision.

    Imaging, radiomic and text payloads are held as references rather than arrays
    so that a cohort can be described without materialising volumes.
    """

    record_id: str
    site: str
    region: str
    stage: StageTriple
    harvested_nodes: int
    scanner_vendor: str
    neoadjuvant_exposed: bool
    lauren: str
    age: int
    sex: str
    ct_ref: str
    consensus_staged: bool = False
    radiomics: np.ndarray | None = None
    clinical: np.ndarray | None = None
    imaging_latent: np.ndarray | None = None
    endoscopy_terms: tuple[str, ...] = ()
    pathology_terms: tuple[str, ...] = ()
    available: dict[str, bool] = field(default_factory=dict)

    def stream_availability(self) -> dict[str, bool]:
        """Which of the five streams were present at the point of care."""
        if self.available:
            return dict(self.available)
        return {
            Stream.CT.value: True,
            Stream.RADIOMICS.value: self.radiomics is not None,
            Stream.CLINICAL.value: self.clinical is not None,
            Stream.ENDOSCOPY.value: bool(self.endoscopy_terms),
            Stream.PATHOLOGY.value: bool(self.pathology_terms),
        }
