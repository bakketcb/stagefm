"""Digests.

Two kinds are computed and they are not interchangeable. A file digest hashes the
bytes on disk and is what the integrity manifest records. A payload digest hashes a
tensor's numeric content and is what a checkpoint comparison uses, because
``torch.save`` embeds storage metadata and a file hash over a ``.pt`` shard changes on
every write even when the tensors are identical.

Ref: Methods Sec. 4.7 (checkpoints and frozen thresholds).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

SKIP_DIRECTORIES = frozenset({"__pycache__", ".git", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".venv"})
DEFAULT_EXCLUDES = ("integrity_manifest.json",)


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """Streamed SHA-256 over a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def payload_digest(arrays: Iterable[Any]) -> str:
    """Digest the numeric content of tensors or arrays, ignoring container layout."""
    digest = hashlib.sha256()
    for item in arrays:
        array = item.detach().cpu().numpy() if hasattr(item, "detach") else np.asarray(item)
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("utf-8"))
        digest.update(str(contiguous.shape).encode("utf-8"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def iter_release_files(root: Path, excluded: Iterable[str] = DEFAULT_EXCLUDES) -> list[Path]:
    """Every committed file below ``root``, skipping caches and named exclusions."""
    skipped = set(excluded)
    found: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRECTORIES for part in path.parts):
            continue
        if path.name in skipped:
            continue
        found.append(path)
    return found


def manifest_digest(root: Path, excluded: Iterable[str] = DEFAULT_EXCLUDES) -> dict[str, Any]:
    """Aggregate digest over sorted (relative path, file SHA-256) pairs."""
    pairs: list[tuple[str, str]] = [(str(path.relative_to(root)), sha256_file(path)) for path in iter_release_files(root, excluded)]
    aggregate = hashlib.sha256()
    for relative, digest in pairs:
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(digest.encode("utf-8"))
    return {
        "file_count": len(pairs),
        "aggregate_sha256": aggregate.hexdigest(),
        "files": {relative: digest for relative, digest in pairs},
    }
