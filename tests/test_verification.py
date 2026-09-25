"""Verification-driver tests.

These tests exercise the driver's own logic and, where they need the shipped
artefacts, they read them rather than rewriting them: a test that called the artefact
writer would replace the canonical report with a partial one produced by whichever
subset of checks the test happened to run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from stagefm.cli.verify import (
    CLAIM_NAME,
    CLAIMS,
    DATASET_URLS_NAME,
    DEVIATIONS,
    MANIFEST_NAME,
    PACKAGE_DIR,
    SUMMARY_NAME,
    VERIFICATION_NAME,
    _brute_force_auc,
    _reference_achievable,
    _SymbolIndex,
    build_claim_document,
    build_report,
    check_claims,
    check_manuscript,
    check_procedures,
    find_repo_root,
    summary_text,
)


def test_repo_root_discovery(repo_root: Path) -> None:
    assert (repo_root / PACKAGE_DIR).is_dir()
    assert find_repo_root(Path(__file__).resolve()) == repo_root


def test_symbol_index_resolves_modules_and_methods(repo_root: Path) -> None:
    index = _SymbolIndex(repo_root / PACKAGE_DIR)
    assert index.resolve("data/schema.py", "StageTriple")
    assert index.resolve("data/staging.py", "treatment_category")
    assert index.resolve("models/stagefm.py", "STAGEFM.forward")
    assert not index.resolve("data/schema.py", "NotAThing")
    assert not index.resolve("does/not/exist.py", "Anything")


def test_every_claim_anchor_resolves(repo_root: Path) -> None:
    claims, check = check_claims(repo_root)
    assert check.status == "PASS", check.detail
    assert check.evidence["unresolved"] == []
    assert check.evidence["claims_without_paper_location"] == []
    assert len(claims) == len(CLAIMS)
    assert check.evidence["anchors"] == sum(len(claim["code"]) for claim in CLAIMS)
    assert all(claim["verification"] == "EXECUTED" for claim in claims)


def test_claim_document_carries_the_deviations(repo_root: Path) -> None:
    claims, _ = check_claims(repo_root)
    document = build_claim_document(claims)
    assert document["artifact"] == "paper-claim to code mapping"
    assert document["venue"] == "npj Digital Medicine"
    assert document["package_dir"] == PACKAGE_DIR
    assert len(document["deviations"]) == len(DEVIATIONS)
    for deviation in document["deviations"]:
        assert deviation["paper_location"] and deviation["deviation"] and deviation["justification"]


def test_independent_procedure_checks_all_pass() -> None:
    results = check_procedures()
    assert results, "the procedure group produced no checks"
    failures = [result for result in results if result.status != "PASS"]
    assert not failures, [f"{result.check_id}: {result.detail}" for result in failures]
    assert len(results) >= 20


def test_manuscript_checks_surface_the_concordance_discrepancy() -> None:
    results = check_manuscript()
    by_id = {result.check_id: result for result in results}
    assert by_id["ms.interaction_discordance"].status == "PASS"
    assert by_id["ms.interaction_ratio_reported"].status == "PASS"
    discrepancy = by_id["ms.interaction_concordance_reported"]
    assert discrepancy.status == "FAIL"
    assert "0.87" in discrepancy.detail
    assert discrepancy.evidence["derived"] == pytest.approx(0.57, abs=0.01)
    assert by_id["ms.variance_reduction"].status == "PASS"
    assert by_id["ms.cohort_arithmetic"].status == "PASS"
    assert by_id["ms.missingness_shares"].status == "PASS"


def test_brute_force_area_matches_a_hand_count() -> None:
    positive = np.array([0.9, 0.4])
    negative = np.array([0.5, 0.2])
    assert _brute_force_auc(positive, negative) == pytest.approx(0.75)


def test_reference_achievable_rule_matches_the_shipped_set() -> None:
    from stagefm.data.schema import ALL_TRIPLES
    from stagefm.data.staging import AchievableSet

    achievable = AchievableSet()
    for stage in ALL_TRIPLES:
        assert achievable.contains(stage) is _reference_achievable(stage.t, stage.n, stage.m)


def test_build_report_status_vocabulary() -> None:
    from stagefm.cli.verify import CheckResult

    root = find_repo_root(Path(__file__).resolve())
    claims, _ = check_claims(root)
    passing = [CheckResult("a", "x", "y", "PASS")]
    assert build_report(root, claims, passing, VERIFICATION_NAME)["summary"]["overall_status"] == "VERIFIED"
    partial = passing + [CheckResult("b", "x", "y", "NOT_RUN")]
    assert build_report(root, claims, partial, VERIFICATION_NAME)["summary"]["overall_status"] == "PARTIALLY_VERIFIED"
    failing = passing + [CheckResult("c", "x", "y", "FAIL")]
    assert build_report(root, claims, failing, VERIFICATION_NAME)["summary"]["overall_status"] == "UNVERIFIED"


def test_summary_text_lists_every_check() -> None:
    from stagefm.cli.verify import CheckResult

    root = find_repo_root(Path(__file__).resolve())
    claims, _ = check_claims(root)
    checks = [CheckResult("one", "x", "first", "PASS"), CheckResult("two", "x", "second", "FAIL", detail="broke")]
    text = summary_text(build_report(root, claims, checks, VERIFICATION_NAME))
    assert "per-check status" in text
    assert "PASS" in text and "FAIL" in text
    assert "one" in text and "two" in text


def test_shipped_report_is_the_complete_one(repo_root: Path) -> None:
    """The report on disk must carry every check the current driver produces.

    A partial report written by a test that invoked the artefact writers would fail
    here, which is the point: the canonical report is the one produced by a full run.
    """
    report_path = repo_root / VERIFICATION_NAME
    assert report_path.is_file(), "the verification report has not been written yet"
    shipped = json.loads(report_path.read_text(encoding="utf-8"))
    assert "checks" in shipped and "summary" in shipped
    assert shipped["summary"]["checks"] == len(shipped["checks"])
    counts = {"PASS": 0, "FAIL": 0, "NOT_RUN": 0, "BLOCKED": 0}
    for check in shipped["checks"]:
        counts[check["status"]] += 1
    assert counts["PASS"] == shipped["summary"]["PASS"]
    assert counts["FAIL"] == shipped["summary"]["FAIL"]
    assert shipped["claim_mapping"]["claims"] == len(CLAIMS)
    assert shipped["summary"]["overall_status"] in {"VERIFIED", "PARTIALLY_VERIFIED", "UNVERIFIED"}
    ids = [check["id"] for check in shipped["checks"]]
    assert len(ids) == len(set(ids)), "check identifiers must be unique"


def test_shipped_artefacts_are_present_and_consistent(repo_root: Path) -> None:
    for name in (CLAIM_NAME, VERIFICATION_NAME, SUMMARY_NAME, MANIFEST_NAME, DATASET_URLS_NAME):
        assert (repo_root / name).is_file(), f"{name} is missing"
    manifest = json.loads((repo_root / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["excluded"] == [MANIFEST_NAME]
    assert MANIFEST_NAME not in manifest["files"]
    assert VERIFICATION_NAME in manifest["files"]
    assert SUMMARY_NAME in manifest["files"]
    assert manifest["file_count"] == len(manifest["files"])
    assert len(manifest["aggregate_sha256"]) == 64


def test_dataset_urls_file_is_tab_separated_and_https(repo_root: Path) -> None:
    lines = [line for line in (repo_root / DATASET_URLS_NAME).read_text(encoding="utf-8").splitlines() if "\t" in line and not line.startswith("#")]
    assert lines, "no dataset links were recorded"
    for line in lines:
        description, _, url = line.partition("\t")
        assert description.strip()
        assert url.strip().startswith("https://")


def test_shipped_artefacts_record_the_release_slug(repo_root: Path) -> None:
    """The report and the manifest name the release slug, never the checkout directory.

    The repository is published under the slug while the local directory is the paper
    title, so a writer that used ``root.name`` would make a clone's artefacts name a
    directory that does not exist and differ in bytes from these.
    """
    from stagefm.version import RELEASE_SLUG

    report = json.loads((repo_root / VERIFICATION_NAME).read_text(encoding="utf-8"))
    manifest = json.loads((repo_root / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert report["release"]["release_slug"] == RELEASE_SLUG
    assert manifest["root"] == RELEASE_SLUG
    assert repo_root.name not in json.dumps([report["release"], manifest["root"]])


def test_manifest_covers_the_source_tree(repo_root: Path) -> None:
    manifest = json.loads((repo_root / MANIFEST_NAME).read_text(encoding="utf-8"))
    files = set(manifest["files"])
    assert any(name.startswith("src/stagefm/") for name in files)
    assert any(name.startswith("configs/experiment/") for name in files)
    assert "README.md" in files
    assert "pyproject.toml" in files
    assert not any("__pycache__" in name for name in files)
    assert not any(name.endswith(".md") for name in files if name != "README.md")
