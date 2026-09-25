"""Release identity.

The release slug is the distribution name, the repository name and the importable
package name: one identifier for all three, so nothing has to derive a name from the
name of the checkout directory. The verification report and the integrity manifest
record the slug rather than the local directory, which is what keeps their bytes
identical in the author's tree, in a clone, and inside the container.

Ref: Methods Sec. 4.7 (results and thresholds are frozen and re-loaded unchanged).
"""

from __future__ import annotations

__all__ = ["RELEASE_SLUG", "__version__"]

RELEASE_SLUG = "stagefm"
__version__ = "0.1.0"
