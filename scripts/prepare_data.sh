#!/usr/bin/env bash
# Build the analysis set.
#
# The clinical cohort is held under a data-usage statement and is not distributed, so
# this script materialises the schema-compatible synthetic cohort that every
# mechanism in the repository is exercised against, and writes its manifest. Point
# COHORT_ROOT at a real de-identified cohort with the same schema to run the
# pipeline on real records instead: the schema contract is in
# src/stagefm/data/schema.py and the manifest fields are listed in the README.
set -euo pipefail

COHORT_ROOT="${COHORT_ROOT:-data}"
RUNS_ROOT="${RUNS_ROOT:-runs}"
PYTHON="${PYTHON:-python3}"

echo "cohort root : ${COHORT_ROOT}"
echo "runs root   : ${RUNS_ROOT}"

"${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

from stagefm.data.cohort import build_site_profiles
from stagefm.data.staging import AchievableSet
from stagefm.data.synthetic import generate_cohort, strata_report
from stagefm.utils.config import CohortConfig, load_yaml
from stagefm.utils.hashing import payload_digest
from stagefm.utils.io import write_json

config = CohortConfig.from_mapping(load_yaml("configs/data/cohort.yaml")["cohort"])
achievable = AchievableSet()
train, internal, external, prospective, profiles = generate_cohort(config, achievable, seed=0)

root = Path(os.environ.get("COHORT_ROOT", "data"))
root.mkdir(parents=True, exist_ok=True)
layers = {"train": train, "internal_test": internal, "external": external, "prospective": prospective}
for name, records in layers.items():
    write_json(
        root / f"{name}.json",
        {
            "records": [
                {
                    "record_id": record.record_id,
                    "site": record.site,
                    "region": record.region,
                    "stage": [record.stage.t, record.stage.n, record.stage.m],
                    "harvested_nodes": record.harvested_nodes,
                    "scanner_vendor": record.scanner_vendor,
                    "neoadjuvant_exposed": record.neoadjuvant_exposed,
                    "lauren": record.lauren,
                    "age": record.age,
                    "sex": record.sex,
                    "ct_ref": record.ct_ref,
                    "available": record.stream_availability(),
                }
                for record in records
            ]
        },
    )

write_json(
    root / "manifest.json",
    {
        "achievable_rule": achievable.rule.value,
        "achievable_size": len(achievable),
        "layer_sizes": {name: len(records) for name, records in layers.items()},
        "sites": [
            {
                "site": profile.site,
                "region": profile.region,
                "median_harvested_nodes": profile.median_harvested_nodes,
                "scanner_vendor": profile.scanner_vendor,
                "development": profile.development,
            }
            for profile in build_site_profiles(config).values()
        ],
        "node_strata": strata_report(train + internal + external, config.adequate_node_yield),
        "descriptor_digest": payload_digest([record.harvested_nodes for record in train[:64]]),
    },
)
print(f"wrote {root}/manifest.json and {len(layers)} layer files")
PY

echo "done"
