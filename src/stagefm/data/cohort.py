"""Cohort layout: sites, regions, the three evaluation layers and the node strata.

The cohort is described here as a contract rather than as data. Sites A-C form the
development layer and are split patient-wise into a training part and an internal
test part; sites D and E are held out in full as the external layer; the
prospective layer is the consecutive run gathered after the model and thresholds
were frozen. Site-level held-out evaluation is used instead of cross-validation
because the claim concerns transport to an unseen site.

Ref: Methods Sec. 4.1 (study design and cohorts); Sec. 4.6 (whole-site split).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..utils.config import CohortConfig
from .schema import Examination

DEFAULT_MEDIAN_NODES = {"A": 32, "B": 27, "C": 21, "D": 18, "E": 16}
DEFAULT_VENDORS = {"A": "Siemens", "B": "GE", "C": "Philips", "D": "Siemens", "E": "Toshiba"}


class Layer(str, Enum):
    """The three mutually exclusive evaluation layers."""

    TRAIN = "train"
    INTERNAL_TEST = "internal_test"
    EXTERNAL = "external"
    PROSPECTIVE = "prospective"


class NodeStratum(str, Enum):
    """Nodal-sampling adequacy stratum."""

    ADEQUATE = "adequate"
    INADEQUATE = "inadequate"


@dataclass(frozen=True)
class SiteProfile:
    """Everything the pipeline needs to know about one contributing site."""

    site: str
    region: str
    median_harvested_nodes: int
    scanner_vendor: str
    development: bool

    @property
    def nodal_ascertainment(self) -> float:
        """Sensitivity of a pathological nodal call at this site.

        Ascertainment rises with the median number of nodes examined, which is the
        property the manuscript's per-site analysis reports (Fig. 2a).
        """
        reference = 32.0
        return float(min(0.99, 0.55 + 0.012 * self.median_harvested_nodes * (32.0 / reference)))


@dataclass(frozen=True)
class CohortSplit:
    """Records grouped by evaluation layer, plus the site profiles behind them."""

    train: tuple[Examination, ...]
    internal_test: tuple[Examination, ...]
    external: tuple[Examination, ...]
    prospective: tuple[Examination, ...]
    profiles: dict[str, SiteProfile]

    def by_layer(self, layer: Layer) -> tuple[Examination, ...]:
        return {
            Layer.TRAIN: self.train,
            Layer.INTERNAL_TEST: self.internal_test,
            Layer.EXTERNAL: self.external,
            Layer.PROSPECTIVE: self.prospective,
        }[layer]

    def sites_in(self, layer: Layer) -> tuple[str, ...]:
        return tuple(sorted({record.site for record in self.by_layer(layer)}))

    def counts(self) -> dict[str, int]:
        return {
            Layer.TRAIN.value: len(self.train),
            Layer.INTERNAL_TEST.value: len(self.internal_test),
            Layer.EXTERNAL.value: len(self.external),
            Layer.PROSPECTIVE.value: len(self.prospective),
        }


def build_site_profiles(config: CohortConfig) -> dict[str, SiteProfile]:
    """Materialise the site table from the cohort config."""
    profiles: dict[str, SiteProfile] = {}
    for site in config.sites:
        profiles[site] = SiteProfile(
            site=site,
            region=config.regions.get(site, "unknown"),
            median_harvested_nodes=int(config.median_harvested_nodes.get(site, 20)),
            scanner_vendor=DEFAULT_VENDORS.get(site, config.scanner_vendors[0]),
            development=site in config.development_sites,
        )
    return profiles


def layer_sizes(config: CohortConfig) -> dict[Layer, int]:
    """Record counts per layer, as declared in the cohort config."""
    return {
        Layer.TRAIN: config.train_size,
        Layer.INTERNAL_TEST: config.internal_test_size,
        Layer.EXTERNAL: config.external_size,
        Layer.PROSPECTIVE: config.prospective_size,
    }


def development_layer_size(config: CohortConfig) -> int:
    """Development records: the sum of the two development-site layers."""
    return config.development_size


def stratum_of(harvested_nodes: int, cutoff: int) -> NodeStratum:
    """Adequacy stratum of one examination against the nodal-yield cutoff."""
    return NodeStratum.ADEQUATE if harvested_nodes >= cutoff else NodeStratum.INADEQUATE


def stratum_indices(records: list[Examination] | tuple[Examination, ...], cutoff: int) -> dict[NodeStratum, list[int]]:
    """Positions of the records in each adequacy stratum."""
    buckets: dict[NodeStratum, list[int]] = {NodeStratum.ADEQUATE: [], NodeStratum.INADEQUATE: []}
    for position, record in enumerate(records):
        buckets[stratum_of(record.harvested_nodes, cutoff)].append(position)
    return buckets


def site_case_shares(records: tuple[Examination, ...]) -> dict[str, float]:
    """Each site's share of a layer, used to check that no site dominates the cohort."""
    total = len(records)
    if total == 0:
        return {}
    tallies: dict[str, int] = {}
    for record in records:
        tallies[record.site] = tallies.get(record.site, 0) + 1
    return {site: count / total for site, count in sorted(tallies.items())}


def subsample_to_cutoff(records: tuple[Examination, ...], cutoff: int) -> tuple[Examination, ...]:
    """Keep only examinations whose nodal yield reaches ``cutoff``."""
    return tuple(record for record in records if record.harvested_nodes >= cutoff)
