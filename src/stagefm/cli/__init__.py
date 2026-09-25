"""Command-line entry points.

``train`` fits one arm on one experiment config, ``eval`` scores a fitted arm on the
external layer, and ``verify`` resolves the paper-claim mapping, executes the
pipeline and writes the release's integrity artefacts.
"""

from __future__ import annotations

__all__ = ["eval", "pipeline", "train", "verify"]
