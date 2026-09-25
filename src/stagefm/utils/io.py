"""Atomic serialisation.

Writers replace their target with ``os.replace`` so a reader never observes a
half-written JSON or text file, and the temporary file is chmod-ed to 0644 before the
replace because ``tempfile.mkstemp`` creates 0600 and would otherwise leave the
release's own artefacts unreadable to anyone but the owner.

Ref: Methods Sec. 4.7 (frozen thresholds and model state are re-loaded unchanged).
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_JSON_KWARGS: dict[str, Any] = {"indent": 2, "sort_keys": False, "ensure_ascii": False}


def _stage_text(text: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.chmod(0o644)
    return tmp


def write_text(path: str | Path, text: str) -> Path:
    """Write text atomically, with a trailing newline guaranteed."""
    target = Path(path)
    body = text if text.endswith("\n") else text + "\n"
    tmp = _stage_text(body, target)
    tmp.replace(target)
    return target


def write_json(path: str | Path, payload: Mapping[str, Any] | Sequence[Any]) -> Path:
    """Serialise ``payload`` to JSON atomically."""
    return write_text(path, json.dumps(payload, **_JSON_KWARGS))


def read_json(path: str | Path) -> Any:
    """Read a JSON document."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def to_builtin(value: Any) -> Any:
    """Recursively convert numpy scalars and arrays into JSON-serialisable values."""
    import numpy as np

    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value
