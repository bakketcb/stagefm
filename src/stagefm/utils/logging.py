"""Logging helpers.

Library modules log through this factory so that the CLI can configure a single
root handler and so that no module writes to stdout directly.

Ref: Methods Sec. 4.7 (training and implementation).
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False
_DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    """Install a single stderr handler on the package root logger.

    Idempotent: repeated calls do not stack handlers, which keeps a resumed run
    from duplicating every line.
    """
    global _CONFIGURED
    root = logging.getLogger("stagefm")
    if _CONFIGURED:
        root.setLevel(level)
        return
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, configured on first use."""
    configure_logging()
    if name.startswith("stagefm"):
        return logging.getLogger(name)
    return logging.getLogger(f"stagefm.{name}")
