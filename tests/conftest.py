"""Shared fixtures.

The smoke pipeline is built once per session because constructing the cohort and the
arm is the expensive part of the suite, and every test that needs a live model shares
the same one. Tests that mutate model weights request the ``fresh_pipeline`` fixture
instead.

The repository root is derived from this file's location so the suite runs from any
working directory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """The release root."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def smoke_config():  # type: ignore[no-untyped-def]
    """The unit-test smoke configuration, resolved through the same loader the CLI uses."""
    from stagefm.utils.config import ExperimentConfig, resolve_experiment

    return ExperimentConfig.from_mapping(resolve_experiment("_smoke", REPO_ROOT / "configs", []))


@pytest.fixture(scope="session")
def smoke_pipeline(smoke_config):  # type: ignore[no-untyped-def]
    """A prepared smoke pipeline, shared across the session."""
    from stagefm.cli.pipeline import prepare

    return prepare(smoke_config, "stagefm")


@pytest.fixture
def fresh_pipeline(smoke_config):  # type: ignore[no-untyped-def]
    """A newly built smoke pipeline, for tests that perturb the weights."""
    from stagefm.cli.pipeline import prepare

    return prepare(smoke_config, "stagefm")


@pytest.fixture
def smoke_batch(smoke_pipeline):  # type: ignore[no-untyped-def]
    """One collated smoke batch drawn from the training layer."""
    from stagefm.data.cohort import Layer
    from stagefm.data.dataset import collate

    return collate([smoke_pipeline.datasets[Layer.TRAIN][index] for index in range(4)])
