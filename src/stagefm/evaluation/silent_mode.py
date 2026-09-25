"""Silent-mode deployment summary.

The model ran non-interventionally across the consecutive examinations: every
eligible examination was scored, the output was available in an optional panel, and
nothing was returned to the treating team. The summaries here are therefore about
deployment cost and consumption rather than about clinical impact, and the panel-open
rate is reported alongside the site's own accuracy because the two are not related
in the way a launch would hope.

Ref: Methods Sec. 2.8 and Table 1 (silent-mode deployment and workflow).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class DeploymentLog:
    """One row per eligible examination."""

    site: np.ndarray
    opened: np.ndarray
    time_to_output_minutes: np.ndarray
    reading_time_seconds: np.ndarray
    unavailable: np.ndarray
    model_minutes: np.ndarray | None = None
    preoperative_discordant: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.site.shape[0])


@dataclass(frozen=True)
class DeploymentSummary:
    """Per-site and overall deployment statistics."""

    per_site: dict[str, dict[str, float]] = field(default_factory=dict)
    overall: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {"per_site": self.per_site, "overall": self.overall}


def _median_iqr(values: np.ndarray) -> tuple[float, float, float]:
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.median(values)),
        float(np.percentile(values, 25)),
        float(np.percentile(values, 75)),
    )


def summarise_deployment(log: DeploymentLog) -> DeploymentSummary:
    """Aggregate the deployment log into the workflow table."""
    per_site: dict[str, dict[str, float]] = {}
    for site in sorted(set(log.site.tolist())):
        selector = log.site == site
        eligible = int(selector.sum())
        opened = int(log.opened[selector].sum())
        median_output, low_output, high_output = _median_iqr(log.time_to_output_minutes[selector])
        median_reading, low_reading, high_reading = _median_iqr(log.reading_time_seconds[selector])
        per_site[str(site)] = {
            "eligible": float(eligible),
            "opened": float(opened),
            "opened_percent": float(100.0 * opened / eligible) if eligible else float("nan"),
            "time_to_output_median_min": median_output,
            "time_to_output_iqr_low": low_output,
            "time_to_output_iqr_high": high_output,
            "reading_time_median_s": median_reading,
            "reading_time_iqr_low": low_reading,
            "reading_time_iqr_high": high_reading,
            "unavailable_percent": float(100.0 * log.unavailable[selector].mean()) if eligible else float("nan"),
        }
    eligible = len(log)
    opened = int(log.opened.sum())
    median_output, low_output, high_output = _median_iqr(log.time_to_output_minutes)
    median_reading, low_reading, high_reading = _median_iqr(log.reading_time_seconds)
    overall = {
        "eligible": float(eligible),
        "opened": float(opened),
        "opened_percent": float(100.0 * opened / eligible) if eligible else float("nan"),
        "time_to_output_median_min": median_output,
        "time_to_output_iqr_low": low_output,
        "time_to_output_iqr_high": high_output,
        "reading_time_median_s": median_reading,
        "reading_time_iqr_low": low_reading,
        "reading_time_iqr_high": high_reading,
        "unavailable_percent": float(100.0 * log.unavailable.mean()) if eligible else float("nan"),
        "unavailable_count": float(log.unavailable.sum()),
        "opened_share_range_low": float(min((row["opened_percent"] for row in per_site.values()), default=float("nan"))),
        "opened_share_range_high": float(max((row["opened_percent"] for row in per_site.values()), default=float("nan"))),
    }
    return DeploymentSummary(per_site=per_site, overall=overall)


def reading_time_distribution(log: DeploymentLog) -> dict[str, float]:
    """Share of interactions under ten seconds and over sixty seconds."""
    values = log.reading_time_seconds
    if values.size == 0:
        return {"under_10s_percent": float("nan"), "over_60s_percent": float("nan")}
    return {
        "under_10s_percent": float(100.0 * (values < 10).mean()),
        "over_60s_percent": float(100.0 * (values > 60).mean()),
    }


def stratified_open_rate(log: DeploymentLog) -> dict[str, dict[str, float]]:
    """Panel-open rate split by whether the preoperative stage was discordant.

    The stratification is the point of the table row: a tool that is used more often
    on the examinations where the clinical stage disagreed with pathology is being
    used where it can help, which is a different claim from being used often.
    """
    if log.preoperative_discordant is None:
        return {}
    flags = log.preoperative_discordant
    out: dict[str, dict[str, float]] = {}
    for label, selector in (("discordant", flags), ("concordant", ~flags)):
        if not selector.any():
            continue
        out[label] = {
            "eligible": float(selector.sum()),
            "opened": float(log.opened[selector].sum()),
            "opened_percent": float(100.0 * log.opened[selector].mean()),
        }
    return out


def timing_composition(log: DeploymentLog) -> dict[str, float]:
    """Share of the median time-to-output spent inside the model."""
    if log.model_minutes is None:
        return {}
    model, _, _ = _median_iqr(log.model_minutes)
    total, _, _ = _median_iqr(log.time_to_output_minutes)
    if not np.isfinite(total) or total <= 0:
        return {"model_minutes": model, "total_minutes": total, "model_share": float("nan")}
    return {
        "model_minutes": model,
        "total_minutes": total,
        "model_share": float(model / total),
        "queue_and_transfer_minutes": float(total - model),
    }


def synthetic_deployment_log(eligible_by_site: dict[str, int], open_rates: dict[str, float], seed: int = 0) -> DeploymentLog:
    """A deployment log with the reported per-site eligibility and open rates.

    The per-site eligibility counts and open rates are cohort-design quantities
    transcribed from the manuscript; the per-examination timings are drawn, so every
    value derived from this log is a generated value and is reported as NOT_RUN.
    """
    rng = np.random.default_rng(seed)
    sites: list[str] = []
    opened: list[bool] = []
    output: list[float] = []
    reading: list[float] = []
    unavailable: list[bool] = []
    model: list[float] = []
    discordant: list[bool] = []
    for site, eligible in eligible_by_site.items():
        rate = open_rates.get(site, 0.78)
        for _ in range(int(eligible)):
            sites.append(site)
            is_discordant = bool(rng.random() < 0.034)
            open_probability = rate + (0.08 if is_discordant else 0.0)
            opened.append(bool(rng.random() < min(open_probability, 1.0)))
            output.append(float(np.clip(rng.normal(6.4, 1.6), 1.0, 20.0)))
            reading.append(float(max(2.0, rng.lognormal(mean=np.log(18.0), sigma=0.7))))
            failed = bool(rng.random() < 0.013)
            unavailable.append(failed)
            model.append(float(np.clip(rng.normal(4.9, 1.1), 0.5, 15.0)))
            discordant.append(is_discordant)
    return DeploymentLog(
        site=np.array(sites, dtype=object),
        opened=np.array(opened, dtype=bool),
        time_to_output_minutes=np.array(output, dtype=np.float64),
        reading_time_seconds=np.array(reading, dtype=np.float64),
        unavailable=np.array(unavailable, dtype=bool),
        model_minutes=np.array(model, dtype=np.float64),
        preoperative_discordant=np.array(discordant, dtype=bool),
    )


def deployment_report(log: DeploymentLog) -> dict[str, object]:
    """Every deployment summary the workflow table needs."""
    return {
        "summary": summarise_deployment(log).as_dict(),
        "reading_time": reading_time_distribution(log),
        "stratified_open_rate": stratified_open_rate(log),
        "timing": timing_composition(log),
    }
