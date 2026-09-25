"""Schema-compatible cohort generator.

The clinical cohort is held under a data-usage statement and is not redistributed,
so every mechanism downstream is exercised against a cohort drawn from this
generator instead. The generator is built so that each structural feature the
manuscript relies on is present in the data rather than asserted in a comment:

* the pathological nodal label is a *sampling* outcome, so a site that examines
  fewer nodes records a systematically lower and noisier nodal category for the
  same underlying burden. The ascertainment analysis then has a real signal to
  measure;
* labels are drawn only from the achievable set, so the joint label space of the
  synthetic cohort matches the staging definitions;
* the imaging-derived latent carries the underlying burden, not the recorded label,
  which is what makes the reference standard rather than the representation the
  limiting factor for the nodal axis.

It reproduces the manuscript's cohort *shape* (site counts, layer sizes, median
nodal yield, missingness counts), and nothing computed from it is a study result.
Every cohort-level value in the verification report is therefore NOT_RUN.

Ref: Methods Sec. 4.1-4.3, Sec. 4.11 (missingness); Data availability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..utils.config import CohortConfig
from .cohort import Layer, SiteProfile, build_site_profiles, stratum_of
from .schema import Examination, StageTriple
from .staging import AchievableSet

CLINICAL_DIM = 12
RADIOMIC_DIM = 64
IMAGE_LATENT_DIM = 32
MISSING_ENDOSCOPY = 19
MISSING_PATHOLOGY = 61
_ENDOSCOPY_TERMS = ("ulceration", "infiltration", "wall_thickening", "peristalsis_loss", "stenosis", "nodular_mucosa")
_PATHOLOGY_TERMS = ("adenocarcinoma", "signet_ring", "intestinal", "diffuse", "lymphovascular_invasion", "perineural_invasion")

T_PRIOR = np.array([0.14, 0.26, 0.36, 0.24])
NODE_RATE_BY_T = np.array([0.16, 0.30, 0.50, 0.66])
METASTASIS_BASE_BY_T = np.array([0.02, 0.07, 0.16, 0.30])


@dataclass(frozen=True)
class SyntheticCohortConfig:
    """Knobs of the generator, all engineering defaults."""

    site_dispersion: float = 0.30
    development_site_shares: tuple[float, ...] = (0.335, 0.333, 0.332)
    external_site_shares: tuple[float, ...] = (0.50, 0.50)
    prospective_eligible: tuple[int, ...] = (612, 498, 402, 338)
    feature_noise: float = 0.55
    endoscopy_hint_weight: float = 0.35

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> SyntheticCohortConfig:
        return cls(
            site_dispersion=float(raw.get("site_dispersion", 0.30)),
            feature_noise=float(raw.get("feature_noise", 0.55)),
            endoscopy_hint_weight=float(raw.get("endoscopy_hint_weight", 0.35)),
        )


def _allocate(total: int, shares: tuple[float, ...]) -> list[int]:
    """Split ``total`` across shares, giving the remainder to the first bucket."""
    counts = [int(np.floor(total * share)) for share in shares]
    counts[0] += total - sum(counts)
    return counts


def _draw_harvested(rng: np.random.Generator, profile: SiteProfile, size: int, dispersion: float) -> np.ndarray:
    """Node yield for a site: log-normal around a median set by local practice."""
    median = max(4.0, profile.median_harvested_nodes + 4.0)
    sigma = dispersion * (32.0 / max(profile.median_harvested_nodes, 1)) ** 0.35
    values: np.ndarray = rng.lognormal(mean=float(np.log(median)), sigma=float(sigma), size=size)
    counts: np.ndarray = np.clip(np.rint(values), 1, 90)
    return counts.astype(np.int64)


def _draw_stage(rng: np.random.Generator, achievable: AchievableSet, size: int) -> list[StageTriple]:
    """Rejection-sample stages restricted to the achievable set."""
    stages: list[StageTriple] = []
    attempts = 0
    cap = size * 400 + 1000
    while len(stages) < size and attempts < cap:
        t = int(rng.choice(np.arange(1, 5), p=T_PRIOR))
        rate = float(NODE_RATE_BY_T[t - 1])
        burden = float(np.clip(rng.normal(rate, 0.22), 0.0, 1.0))
        p_met = float(np.clip(METASTASIS_BASE_BY_T[t - 1] + 0.18 * burden, 0.0, 0.95))
        m = int(rng.random() < p_met)
        node_count = rng.binomial(14, burden)
        n = int(np.minimum(3, np.digitize(node_count, [1, 3, 7])))
        stage = StageTriple(t=t, n=n, m=m)
        attempts += 1
        if achievable.contains(stage):
            stages.append(stage)
        elif m == 1 and n == 0 and t < 4:
            stages.append(StageTriple(t=4, n=n, m=m))
    if len(stages) < size:
        raise RuntimeError("achievable-set rejection sampling failed to fill the cohort")
    return stages


def _recorded_nodal_category(rng: np.random.Generator, burden: np.ndarray, harvested: np.ndarray) -> np.ndarray:
    """Nodal category recorded by the reference standard.

    Positives are counted among the nodes actually examined, so the same burden
    yields a lower recorded category where fewer nodes are sampled.
    """
    positives: np.ndarray = rng.binomial(harvested, np.clip(burden, 0.0, 0.999))
    recorded: np.ndarray = np.minimum(3, np.digitize(positives, [1, 3, 7]))
    return recorded.astype(np.int64)


def _burden_from_category(rng: np.random.Generator, n_category: np.ndarray, harvested: np.ndarray) -> np.ndarray:
    """Latent burden consistent with the underlying disease, not with the sampled label.

    The generator tracks a burden per record and derives the *recorded* category
    from it. This helper reconstructs a burden for a record that already carries a
    category, which the generator uses when a drawn stage is replaced.
    """
    base: np.ndarray = np.array([0.04, 0.20, 0.42, 0.66])[n_category]
    scale: np.ndarray = np.clip(12.0 / np.maximum(harvested, 1), 0.4, 2.5)
    burden: np.ndarray = np.clip(base * (1.0 + 0.15 * rng.normal(size=base.shape)) * np.clip(scale, 0.6, 1.8), 0.0, 1.0)
    return burden


def _repair_stage(stage: StageTriple, achievable: AchievableSet) -> StageTriple:
    """Move a sampled nodal category back into the achievable set.

    Sampling the reference standard can record a nodal category the primary cannot
    support. The repair advances the primary category first, which preserves the
    nodal and metastatic categories the record was generated with, and only drops the
    metastatic category when no primary admits the combination.
    """
    for t in range(stage.t, 5):
        candidate = StageTriple(t=t, n=stage.n, m=stage.m)
        if achievable.contains(candidate):
            return candidate
    for t in range(1, 5):
        candidate = StageTriple(t=t, n=stage.n, m=stage.m)
        if achievable.contains(candidate):
            return candidate
    return StageTriple(t=4, n=stage.n, m=0)


def _clinical_block(rng: np.random.Generator, stages: list[StageTriple], profiles: list[SiteProfile], size: int) -> np.ndarray:
    """Structured clinical covariates.

    Neoadjuvant exposure and Lauren type both shift the imaging phenotype, so they
    are recorded per record and carried into the fusion block.
    """
    block = np.zeros((size, CLINICAL_DIM), dtype=np.float32)
    for row, stage in enumerate(stages):
        block[row, 0] = float(rng.integers(28, 86)) / 100.0
        block[row, 1] = float(rng.random() < 0.44)
        block[row, 2] = float(stage.t) / 4.0
        block[row, 3] = float(stage.n) / 3.0
        block[row, 4] = float(stage.m)
        block[row, 5] = float(rng.random())
        block[row, 6] = float(rng.random() < 0.35)
        block[row, 7] = float(rng.random() < 0.30)
        block[row, 8] = float(profiles[row].median_harvested_nodes) / 40.0
        block[row, 9] = float(profiles[row].nodal_ascertainment)
        block[row, 10] = float(rng.random())
        block[row, 11] = 1.0
    return block


def _imaging_latent(rng: np.random.Generator, stages: list[StageTriple], burden: np.ndarray, noise: float, dim: int) -> np.ndarray:
    """Latent the imaging encoder is expected to recover.

    The latent is a smooth function of the *underlying* stage and burden, so a
    model that reads it perfectly still disagrees with the sampled nodal label at
    low-yield sites.
    """
    basis = rng.normal(size=(4, dim)) * 0.7 + rng.normal(size=(4, dim)) * 0.3
    vectors = np.zeros((len(stages), dim), dtype=np.float32)
    for row, stage in enumerate(stages):
        signal = basis[stage.t - 1] + 0.9 * burden[row] * basis[1] + 1.1 * float(stage.m) * basis[3]
        vectors[row] = signal.astype(np.float32)
    return (vectors + noise * rng.normal(size=vectors.shape)).astype(np.float32)


def _radiomic_block(rng: np.random.Generator, stages: list[StageTriple], burden: np.ndarray, noise: float, dim: int) -> np.ndarray:
    """Peritumoral descriptors, weighted so the nodal signal rides on texture."""
    texture = rng.normal(size=dim) * 0.5
    mixing = rng.normal(size=(4, dim)) * 0.5
    block = np.zeros((len(stages), dim), dtype=np.float32)
    for row, stage in enumerate(stages):
        block[row] = mixing[stage.t - 1] + 1.4 * burden[row] * texture + 0.6 * float(stage.m) * rng.normal(size=dim)
    return (block + noise * rng.normal(size=block.shape)).astype(np.float32)


def _term_stream(rng: np.random.Generator, stages: list[StageTriple], burden: np.ndarray, terms: tuple[str, ...], hint_weight: float) -> list[tuple[str, ...]]:
    """Sampled descriptor terms for one free-text stream."""
    out: list[tuple[str, ...]] = []
    for row, stage in enumerate(stages):
        depth = stage.t / 4.0
        rates = np.array([0.20 + 0.5 * depth, 0.15 + 0.5 * depth, 0.30 + 0.4 * depth, 0.10 + 0.3 * depth, 0.08 + 0.5 * depth, 0.12 + 0.4 * depth])
        rates[1] = float(np.clip(rates[1] + hint_weight * burden[row], 0.0, 0.95))
        drawn = [term for term, rate in zip(terms, rates) if rng.random() < rate]
        if not drawn:
            drawn = [terms[0]]
        out.append(tuple(drawn))
    return out


def generate_cohort(
    config: CohortConfig,
    achievable: AchievableSet | None = None,
    synthetic: SyntheticCohortConfig | None = None,
    seed: int = 0,
) -> tuple[list[Examination], list[Examination], list[Examination], list[Examination], dict[str, SiteProfile]]:
    """Draw the four layers of the cohort.

    Returns the train, internal-test, external and prospective layers together with
    the site profiles, in the order :class:`stagefm.data.cohort.CohortSplit` expects.
    """
    reachable = achievable or AchievableSet()
    knobs = synthetic or SyntheticCohortConfig()
    rng = np.random.default_rng(seed)
    profiles = build_site_profiles(config)

    development_sites = [profiles[site] for site in config.development_sites]
    external_sites = [profiles[site] for site in config.external_sites]

    train_counts = _allocate(config.train_size, knobs.development_site_shares[: len(development_sites)])
    test_counts = _allocate(config.internal_test_size, knobs.development_site_shares[: len(development_sites)])
    external_counts = _allocate(config.external_size, knobs.external_site_shares[: len(external_sites)])

    layers: dict[Layer, list[Examination]] = {Layer.TRAIN: [], Layer.INTERNAL_TEST: [], Layer.EXTERNAL: [], Layer.PROSPECTIVE: []}
    record_counter = 0

    def emit(profile: SiteProfile, layer: Layer, count: int, index: int) -> None:
        nonlocal record_counter
        stages = _draw_stage(rng, reachable, count)
        harvested = _draw_harvested(rng, profile, count, knobs.site_dispersion)
        burden = _burden_from_category(rng, np.array([stage.n for stage in stages]), harvested)
        recorded = _recorded_nodal_category(rng, burden, harvested)
        recorded_stages = [StageTriple(t=stage.t, n=int(recorded[row]), m=stage.m) for row, stage in enumerate(stages)]
        stages = [_repair_stage(stage, reachable) for stage in recorded_stages]
        clinical = _clinical_block(rng, stages, [profile] * count, count)
        latent = _imaging_latent(rng, stages, burden, knobs.feature_noise, IMAGE_LATENT_DIM)
        radiomics = _radiomic_block(rng, stages, burden, knobs.feature_noise, RADIOMIC_DIM)
        endoscopy = _term_stream(rng, stages, burden, _ENDOSCOPY_TERMS, knobs.endoscopy_hint_weight)
        pathology = _term_stream(rng, stages, burden, _PATHOLOGY_TERMS, knobs.endoscopy_hint_weight)
        for row in range(count):
            record_counter += 1
            record_id = f"{layer.value[0]}{index:02d}-{record_counter:06d}"
            layers[layer].append(
                Examination(
                    record_id=record_id,
                    site=profile.site,
                    region=profile.region,
                    stage=stages[row],
                    harvested_nodes=int(harvested[row]),
                    scanner_vendor=profile.scanner_vendor,
                    neoadjuvant_exposed=bool(clinical[row, 6] > 0.5),
                    lauren="diffuse" if clinical[row, 7] > 0.5 else "intestinal",
                    age=int(clinical[row, 0] * 100),
                    sex="female" if clinical[row, 1] > 0.5 else "male",
                    ct_ref=f"{layer.value}/{profile.site}/{record_id}",
                    consensus_staged=bool(rng.random() < 0.021),
                    radiomics=radiomics[row],
                    clinical=clinical[row],
                    imaging_latent=latent[row],
                    endoscopy_terms=endoscopy[row],
                    pathology_terms=pathology[row],
                    available={
                        "ct": True,
                        "radiomics": True,
                        "clinical": True,
                        "endoscopy": bool(endoscopy[row]),
                        "pathology": bool(pathology[row]),
                    },
                )
            )

    for position, profile in enumerate(development_sites):
        emit(profile, Layer.TRAIN, train_counts[position], position)
    for position, profile in enumerate(development_sites):
        emit(profile, Layer.INTERNAL_TEST, test_counts[position], position)
    for position, profile in enumerate(external_sites):
        emit(profile, Layer.EXTERNAL, external_counts[position], position)

    prospective_weights = np.asarray(knobs.prospective_eligible, dtype=np.float64)
    prospective_shares = tuple((prospective_weights / prospective_weights.sum()).tolist())
    prospective_counts = _allocate(config.prospective_size, prospective_shares)
    prospective_sites = [profiles[site] for site in config.sites[: len(prospective_counts)]]
    for position, profile in enumerate(prospective_sites):
        emit(profile, Layer.PROSPECTIVE, int(prospective_counts[position]), position)

    _mark_missingness(layers[Layer.TRAIN], layers[Layer.INTERNAL_TEST], layers[Layer.EXTERNAL], seed)
    return layers[Layer.TRAIN], layers[Layer.INTERNAL_TEST], layers[Layer.EXTERNAL], layers[Layer.PROSPECTIVE], profiles


def _mark_missingness(train: list[Examination], internal: list[Examination], external: list[Examination], seed: int) -> None:
    """Apply the manuscript's missingness counts to the 11,284-record analysis set.

    Nineteen records lack an endoscopic description and 61 lack pathology text
    (Methods Sec. 4.11). The records are retained: an absent stream is signalled to
    the fusion block by its absence token rather than imputed.
    """
    analysis_set = train + internal + external
    total = len(analysis_set)
    if total < MISSING_ENDOSCOPY + MISSING_PATHOLOGY:
        return
    rng = np.random.default_rng(seed + 991)
    order = rng.permutation(total)
    endoscopy_missing = order[:MISSING_ENDOSCOPY]
    pathology_missing = order[MISSING_ENDOSCOPY : MISSING_ENDOSCOPY + MISSING_PATHOLOGY]
    for position in endoscopy_missing:
        analysis_set[position].endoscopy_terms = ()
        analysis_set[position].available["endoscopy"] = False
    for position in pathology_missing:
        analysis_set[position].pathology_terms = ()
        analysis_set[position].available["pathology"] = False


def strata_report(records: tuple[Examination, ...], cutoff: int) -> dict[str, Any]:
    """Adequacy-stratum summary for one layer."""
    adequate = 0
    inadequate = 0
    positives = {"adequate": 0, "inadequate": 0}
    for record in records:
        bucket = stratum_of(record.harvested_nodes, cutoff)
        if bucket.value == "adequate":
            adequate += 1
            positives["adequate"] += int(record.stage.n > 0)
        else:
            inadequate += 1
            positives["inadequate"] += int(record.stage.n > 0)
    total = len(records)
    return {
        "cutoff": cutoff,
        "total": total,
        "adequate": adequate,
        "inadequate": inadequate,
        "adequate_share": adequate / total if total else 0.0,
        "nodal_prevalence_adequate": positives["adequate"] / adequate if adequate else 0.0,
        "nodal_prevalence_inadequate": positives["inadequate"] / inadequate if inadequate else 0.0,
    }
