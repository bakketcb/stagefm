"""Stage-consistent multimodal radiomics foundation model for gastric cancer staging.

This package holds the model, data, training and evaluation code behind the
manuscript's staging system. The four modules named in Methods Sec. 4.5 -- the
frozen abdominal-CT vision-language encoder with low-rank adaptation, the
cross-attention fusion block, the stage-consistency block and the site-conditional
risk-control layer -- live under :mod:`stagefm.models`; the cohort contract, the
radiomic descriptor pipeline and the modality streams live under
:mod:`stagefm.data`.
"""

from __future__ import annotations

from .version import RELEASE_SLUG, __version__

__all__ = ["RELEASE_SLUG", "__version__"]
