"""Reader study aggregation.

The reader study is paired within reader and within case: every reader sees every
selected examination twice, once without and once with the model's output, with the
unassisted session first and a washout interval between them. The aggregation
therefore has to keep the pairing, which is why every comparison here works on the
per-observation records rather than on reader-averaged summaries.

The response table is not part of this release, so the aggregation functions run
against whatever table they are given and the reported values stay NOT_RUN unless a
real table is supplied.

Ref: Methods Sec. 4.9 (reader study design); Sec. 2.6 and Sec. 3 (assisted reading
result).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..metrics.concordance import axis_agreement, weighted_kappa
from ..metrics.decision import discordance_summary
from ..stats.bootstrap import percentile_interval

READER_COUNT = 18
EXAM_COUNT = 360
EXAMS_PER_SITE = 72
WASHOUT_DAYS = 14


@dataclass(frozen=True)
class ReaderStudyDesign:
    """The pre-specified design of the reader study."""

    readers: int = READER_COUNT
    exams: int = EXAM_COUNT
    exams_per_site: int = EXAMS_PER_SITE
    sites: tuple[str, ...] = ("A", "B", "C", "D", "E")
    washout_days: int = WASHOUT_DAYS

    @property
    def paired_observations(self) -> int:
        """Total paired observations: readers times examinations."""
        return self.readers * self.exams

    def as_dict(self) -> dict[str, object]:
        return {
            "readers": self.readers,
            "exams": self.exams,
            "exams_per_site": self.exams_per_site,
            "sites": list(self.sites),
            "washout_days": self.washout_days,
            "paired_observations": self.paired_observations,
            "expected_span": self.exams_per_site * len(self.sites),
        }

    def validate(self) -> list[str]:
        """Consistency checks on the design as stated."""
        problems: list[str] = []
        if self.exams_per_site * len(self.sites) != self.exams:
            problems.append("exams per site times sites does not equal the reading-set size")
        if self.readers % 2 != 0:
            problems.append("reader count is odd, so the radiologist/clinician split is unequal")
        return problems


@dataclass(frozen=True)
class ReaderTable:
    """Reader responses with the pairing preserved."""

    reader_id: np.ndarray
    exam_id: np.ndarray
    site: np.ndarray
    assisted: np.ndarray
    predicted_t: np.ndarray
    reference_t: np.ndarray
    predicted_column: np.ndarray
    reference_column: np.ndarray
    elapsed_days: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.reader_id.shape[0])

    def arm(self, assisted: bool) -> ReaderTable:
        selector = self.assisted == assisted
        return ReaderTable(
            reader_id=self.reader_id[selector],
            exam_id=self.exam_id[selector],
            site=self.site[selector],
            assisted=self.assisted[selector],
            predicted_t=self.predicted_t[selector],
            reference_t=self.reference_t[selector],
            predicted_column=self.predicted_column[selector],
            reference_column=self.reference_column[selector],
            elapsed_days=None if self.elapsed_days is None else self.elapsed_days[selector],
        )

    def validate_pairing(self, washout_days: int = WASHOUT_DAYS) -> list[str]:
        """Pairing and washout checks, returned as a list of problems."""
        problems: list[str] = []
        unassisted = self.arm(False)
        assisted = self.arm(True)
        if len(unassisted) != len(assisted):
            problems.append("the two arms do not contain the same number of observations")
        if set(map(tuple, np.stack([unassisted.reader_id, unassisted.exam_id], axis=1))) != set(
            map(tuple, np.stack([assisted.reader_id, assisted.exam_id], axis=1))
        ):
            problems.append("the two arms are not paired on reader and examination")
        if self.elapsed_days is not None and self.elapsed_days.size and float(self.elapsed_days.min()) < washout_days:
            problems.append("the washout interval between sessions is shorter than the pre-specified one")
        return problems


@dataclass(frozen=True)
class ReaderPerformance:
    """Per-reader accuracy, agreement and discordance for one arm."""

    reader_id: int
    count: int
    t_accuracy: float
    weighted_kappa: float
    discordance_percent: float

    def as_dict(self) -> dict[str, object]:
        return {
            "reader_id": self.reader_id,
            "count": self.count,
            "t_accuracy": self.t_accuracy,
            "weighted_kappa": self.weighted_kappa,
            "discordance_percent": self.discordance_percent,
        }


def reader_performance(table: ReaderTable) -> list[ReaderPerformance]:
    """One row per reader."""
    rows: list[ReaderPerformance] = []
    for reader in sorted(set(table.reader_id.tolist())):
        selector = table.reader_id == reader
        rows.append(
            ReaderPerformance(
                reader_id=int(reader),
                count=int(selector.sum()),
                t_accuracy=axis_agreement(table.predicted_t[selector], table.reference_t[selector]),
                weighted_kappa=weighted_kappa(table.predicted_column[selector], table.reference_column[selector]),
                discordance_percent=discordance_summary(table.predicted_column[selector], table.reference_column[selector], list(table.site[selector])).pooled,
            )
        )
    return rows


def arm_summary(table: ReaderTable) -> dict[str, float]:
    """Pooled accuracy, agreement and discordance for one arm."""
    return {
        "count": float(len(table)),
        "t_accuracy": float(axis_agreement(table.predicted_t, table.reference_t)) if len(table) else float("nan"),
        "weighted_kappa": weighted_kappa(table.predicted_column, table.reference_column),
        "discordance_percent": discordance_summary(table.predicted_column, table.reference_column, list(table.site)).pooled,
    }


def paired_reader_difference(
    unassisted: ReaderTable, assisted: ReaderTable, metric: str = "accuracy", resamples: int = 2000, seed: int = 0
) -> dict[str, float]:
    """Paired bootstrap of the assisted-minus-unassisted difference.

    The resample unit is the paired observation, so the same reader-case pair enters
    both arms in every replicate.
    """
    if len(unassisted) != len(assisted):
        raise ValueError("arms must be paired and of equal length")
    if metric == "accuracy":
        difference = (assisted.predicted_t == assisted.reference_t).astype(np.float64) - (unassisted.predicted_t == unassisted.reference_t).astype(np.float64)
    elif metric == "discordance":
        from ..metrics.decision import discordance_indicator

        difference = discordance_indicator(unassisted.predicted_column, unassisted.reference_column).astype(np.float64) - discordance_indicator(
            assisted.predicted_column, assisted.reference_column
        ).astype(np.float64)
    else:
        raise ValueError(f"unsupported reader-study metric: {metric}")
    count = difference.size
    if count == 0:
        return {"difference": float("nan"), "low": float("nan"), "high": float("nan")}
    rng = np.random.default_rng(seed)
    replicates = np.empty(resamples, dtype=np.float64)
    for position in range(resamples):
        draw = rng.integers(0, count, size=count)
        replicates[position] = difference[draw].mean()
    low, high = percentile_interval(replicates, 0.05)
    return {
        "difference": float(difference.mean()),
        "low": low,
        "high": high,
        "replicates": float(resamples),
    }


def reader_site_breakdown(table: ReaderTable) -> dict[str, dict[str, float]]:
    """Accuracy and discordance per site, for the design's site balance."""
    out: dict[str, dict[str, float]] = {}
    for site in sorted(set(table.site.tolist())):
        selector = table.site == site
        sub = ReaderTable(
            reader_id=table.reader_id[selector],
            exam_id=table.exam_id[selector],
            site=table.site[selector],
            assisted=table.assisted[selector],
            predicted_t=table.predicted_t[selector],
            reference_t=table.reference_t[selector],
            predicted_column=table.predicted_column[selector],
            reference_column=table.reference_column[selector],
        )
        out[str(site)] = arm_summary(sub)
    return out


@dataclass
class ReaderStudyReport:
    """Assembled reader-study report."""

    design: ReaderStudyDesign
    problems: list[str] = field(default_factory=list)
    unassisted: dict[str, float] = field(default_factory=dict)
    assisted: dict[str, float] = field(default_factory=dict)
    accuracy_difference: dict[str, float] = field(default_factory=dict)
    discordance_difference: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "design": self.design.as_dict(),
            "problems": self.problems,
            "unassisted": self.unassisted,
            "assisted": self.assisted,
            "accuracy_difference": self.accuracy_difference,
            "discordance_difference": self.discordance_difference,
        }


def reader_study_report(table: ReaderTable, design: ReaderStudyDesign | None = None) -> ReaderStudyReport:
    """Assemble the reader-study report from a response table."""
    chosen = design or ReaderStudyDesign()
    unassisted = table.arm(False)
    assisted = table.arm(True)
    return ReaderStudyReport(
        design=chosen,
        problems=chosen.validate() + table.validate_pairing(chosen.washout_days),
        unassisted=arm_summary(unassisted),
        assisted=arm_summary(assisted),
        accuracy_difference=paired_reader_difference(unassisted, assisted, "accuracy"),
        discordance_difference=paired_reader_difference(unassisted, assisted, "discordance"),
    )
