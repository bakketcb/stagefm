"""Automated localisation of the stomach and the perigastric nodal basin.

The manuscript relies on no manual lesion definition: the gastric wall and the
nodal-basin boundary are localised automatically and the same stage is applied
unchanged to every site. The localisation here is intensity- and morphology-based
on the preprocessed volume: a soft-tissue band is thresholded, the gastric lumen
and wall are separated by size and elongation, and the basin is the region
immediately superior, posterior and left of the stomach.

The peritumoral descriptor shell is the region between ``shell_inner_mm`` and
``shell_outer_mm`` outside the localised wall, so a descriptor describes the tissue
immediately surrounding the tumour rather than the tumour interior.

Ref: Methods Sec. 4.3 (segmentation and the 0-5 mm peritumoral shell); Sec. 4.4
(organ localisation in preprocessing).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class OrganMasks:
    """The masks produced for one examination."""

    stomach: np.ndarray
    wall: np.ndarray
    nodal_basin: np.ndarray
    peritumoral_shell: np.ndarray

    def descriptor_mask(self) -> np.ndarray:
        """Voxels the radiomic descriptors are computed over."""
        return self.peritumoral_shell


def _soft_tissue_band(volume: np.ndarray) -> np.ndarray:
    """Body voxels whose intensity sits in the soft-tissue range of the normalised volume."""
    body = volume > np.percentile(volume, 35)
    soft = (volume > 0.45) & (volume < 0.85)
    opened: np.ndarray = ndimage.binary_opening(body & soft, structure=np.ones((3, 3, 3)))
    return opened


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Largest 26-connected component of a binary mask."""
    labelled, count = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    if count == 0:
        return mask.astype(bool)
    sizes = ndimage.sum_labels(np.ones_like(labelled), labelled, index=np.arange(1, count + 1))
    return np.asarray(labelled == int(np.argmax(sizes)) + 1, dtype=bool)


def _gastric_lumen(volume: np.ndarray) -> np.ndarray:
    """Low-attenuation gastric content, used as the anchor for the wall.

    The lumen is the largest air- or fluid-filled component enclosed by soft tissue
    in the upper abdomen. To avoid the colon and the small bowel, the component is
    required to be internal: it must not touch the cranial-most slice.
    """
    cavity = volume < 0.22
    cavity = ndimage.binary_opening(cavity, structure=np.ones((3, 3, 3)))
    interior = cavity.copy()
    interior[:2, :, :] = False
    interior[:, :2, :] = False
    interior[:, :, :2] = False
    interior[:, -2:, :] = False
    interior[:, :, -2:] = False
    if not interior.any():
        return largest_component(cavity)
    return largest_component(interior)


def localise_stomach(volume: np.ndarray, spacing_mm: tuple[float, float, float]) -> np.ndarray:
    """Localise the gastric wall as a thin rim around the lumen."""
    lumen = _gastric_lumen(volume)
    rim_mm = 6.0
    dilation_voxels = tuple(max(int(round(rim_mm / max(step, 1e-3))), 1) for step in spacing_mm)
    structure = np.zeros((2 * dilation_voxels[0] + 1, 2 * dilation_voxels[1] + 1, 2 * dilation_voxels[2] + 1), dtype=bool)
    grid = np.ogrid[tuple(slice(-n, n + 1) for n in dilation_voxels)]
    radius = np.sqrt(sum((axis / max(step, 1e-3)) ** 2 for axis, step in zip(grid, spacing_mm)))
    structure[radius <= rim_mm] = True
    dilated: np.ndarray = ndimage.binary_dilation(lumen, structure=structure)
    wall = dilated & ~lumen & _soft_tissue_band(volume)
    if not wall.any():
        wall = dilated & ~lumen
    closed: np.ndarray = ndimage.binary_closing(wall, structure=np.ones((3, 3, 3)))
    return largest_component(closed)


def nodal_basin_mask(volume: np.ndarray, wall: np.ndarray, spacing_mm: tuple[float, float, float], reach_mm: float = 40.0) -> np.ndarray:
    """Perigastric nodal basin: the soft-tissue region draining the stomach.

    The basin is grown from the wall over soft tissue up to ``reach_mm``, excluding
    the wall itself and the gastric lumen, with the reach shortened in the
    anterior-lateral direction where the basin is not defined.
    """
    soft = _soft_tissue_band(volume)
    distance_mm = distance_map_mm(~wall, spacing_mm)
    reach = distance_mm <= reach_mm
    depth, height, width = wall.shape
    anterior_limit = int(0.25 * depth)
    reach[:anterior_limit, :, :] = False
    return np.asarray(reach & soft, dtype=bool)


def distance_map_mm(mask: np.ndarray, spacing_mm: tuple[float, float, float]) -> np.ndarray:
    """Euclidean distance in millimetres from every voxel to the nearest True voxel of ``mask``."""
    if not mask.any():
        return np.full(mask.shape, np.inf, dtype=np.float64)
    distance: np.ndarray = ndimage.distance_transform_edt(~mask, sampling=spacing_mm)
    return distance.astype(np.float64)


def peritumoral_shell(
    wall: np.ndarray,
    spacing_mm: tuple[float, float, float],
    inner_mm: float,
    outer_mm: float,
) -> np.ndarray:
    """Region between ``inner_mm`` and ``outer_mm`` outside the localised wall."""
    if outer_mm <= inner_mm:
        raise ValueError("shell outer radius must exceed its inner radius")
    distance = distance_map_mm(wall, spacing_mm)
    return np.asarray((distance > inner_mm) & (distance <= outer_mm), dtype=bool)


def localise(volume: np.ndarray, spacing_mm: tuple[float, float, float], inner_mm: float, outer_mm: float) -> OrganMasks:
    """Run the full localisation stage for one preprocessed volume."""
    wall = localise_stomach(volume, spacing_mm)
    basin = nodal_basin_mask(volume, wall, spacing_mm)
    shell = peritumoral_shell(wall, spacing_mm, inner_mm, outer_mm)
    if not shell.any():
        shell = np.asarray(basin, dtype=bool)
    return OrganMasks(stomach=wall | basin, wall=wall, nodal_basin=basin, peritumoral_shell=shell)


def stability_perturbations(mask: np.ndarray, repeats: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Independently repeated segmentations used by the stability criterion.

    Each repeat moves the boundary by up to one voxel along a randomly chosen axis
    and direction, which is the segmentation variability the descriptor filter has
    to survive.
    """
    perturbations: list[np.ndarray] = []
    for _ in range(repeats):
        axis = int(rng.integers(0, 3))
        direction = 1 if rng.random() < 0.5 else -1
        perturbed = np.roll(mask, direction, axis=axis)
        if direction > 0:
            perturbed = perturbed | mask
            selector = [slice(None)] * 3
            selector[axis] = slice(0, 1)
            perturbed[tuple(selector)] = False
        else:
            perturbed = perturbed & mask
        if not perturbed.any():
            perturbed = mask.copy()
        perturbations.append(np.asarray(perturbed, dtype=bool))
    return perturbations
