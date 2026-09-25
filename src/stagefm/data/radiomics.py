"""Peritumoral radiomic descriptors, stability selection and redundancy collapse.

Descriptors are computed on the intensity volume restricted to the peritumoral
shell. Definitions follow the standardised formulations the manuscript points to
through the CLEAR checklist rather than private conventions, so the descriptor set
is comparable with the wider radiomics literature:

* first-order statistics over the shell intensities;
* shape descriptors of the shell geometry in millimetres;
* texture matrices (GLCM, GLRLM, GLSZM, GLDM, NGTDM) on a fixed-bin quantisation.

Two properties are handled explicitly rather than assumed. A descriptor must
survive repeated segmentations, which is checked with an intraclass correlation
across independently perturbed masks; and correlated descriptors are collapsed to
one representative per block rather than entered jointly. Both filters are fitted
on the development split alone, so the external sites stay out of the descriptor
selection step as well as out of training.

Descriptors are z-scored within each site using development-set statistics only.

Ref: Methods Sec. 4.3 (radiomic descriptors, stability criterion, redundancy,
per-site z-scoring); Ref. [53] (CLEAR), Ref. [56] (feature reproducibility),
Ref. [57] (multicollinearity).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..utils.config import RadiomicsConfig

QUANTISATION_BINS = 32
_DIRECTIONS: tuple[tuple[int, int, int], ...] = tuple(
    (dz, dy, dx)
    for dz in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if not (dz == 0 and dy == 0 and dx == 0) and (dz > 0 or (dz == 0 and (dy > 0 or (dy == 0 and dx > 0))))
)

FIRST_ORDER_NAMES = (
    "fo_mean",
    "fo_variance",
    "fo_skewness",
    "fo_kurtosis",
    "fo_median",
    "fo_minimum",
    "fo_p10",
    "fo_p90",
    "fo_range",
    "fo_interquartile_range",
    "fo_mean_absolute_deviation",
    "fo_robust_mean_absolute_deviation",
    "fo_energy",
    "fo_root_mean_square",
    "fo_entropy",
    "fo_uniformity",
    "fo_coefficient_of_variation",
    "fo_median_absolute_deviation",
)
SHAPE_NAMES = (
    "sh_volume_mm3",
    "sh_surface_area_mm2",
    "sh_surface_to_volume",
    "sh_sphericity",
    "sh_max_diameter_mm",
    "sh_major_axis_mm",
    "sh_minor_axis_mm",
    "sh_elongation",
    "sh_flatness",
)
GLCM_NAMES = (
    "glcm_contrast",
    "glcm_dissimilarity",
    "glcm_homogeneity",
    "glcm_energy",
    "glcm_correlation",
    "glcm_angular_second_moment",
    "glcm_entropy",
    "glcm_max_probability",
    "glcm_inverse_difference_moment",
    "glcm_maximal_correlation",
)
GLRLM_NAMES = (
    "glrlm_short_run_emphasis",
    "glrlm_long_run_emphasis",
    "glrlm_grey_level_non_uniformity",
    "glrlm_run_length_non_uniformity",
    "glrlm_run_percentage",
    "glrlm_low_grey_level_run_emphasis",
    "glrlm_high_grey_level_run_emphasis",
    "glrlm_grey_level_variance",
)
GLSZM_NAMES = (
    "glszm_small_area_emphasis",
    "glszm_large_area_emphasis",
    "glszm_grey_level_non_uniformity",
    "glszm_size_zone_non_uniformity",
    "glszm_zone_percentage",
    "glszm_low_grey_level_zone_emphasis",
    "glszm_high_grey_level_zone_emphasis",
    "glszm_size_zone_variance",
)
GLDM_NAMES = (
    "gldm_small_dependence_emphasis",
    "gldm_large_dependence_emphasis",
    "gldm_grey_level_non_uniformity",
    "gldm_dependence_non_uniformity",
    "gldm_dependence_entropy",
)
NGTDM_NAMES = (
    "ngtdm_coarseness",
    "ngtdm_contrast",
    "ngtdm_busyness",
    "ngtdm_complexity",
    "ngtdm_strength",
)

FAMILY_NAMES: dict[str, tuple[str, ...]] = {
    "firstorder": FIRST_ORDER_NAMES,
    "shape": SHAPE_NAMES,
    "glcm": GLCM_NAMES,
    "glrlm": GLRLM_NAMES,
    "glszm": GLSZM_NAMES,
    "gldm": GLDM_NAMES,
    "ngtdm": NGTDM_NAMES,
}


@dataclass(frozen=True)
class DescriptorSet:
    """A named descriptor vector with its family labels."""

    names: tuple[str, ...]
    values: np.ndarray

    def as_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in zip(self.names, self.values)}


def _axis_selector(axis: int, value: Any) -> list[Any]:
    """A three-axis slicing tuple with one axis replaced by ``value``."""
    selector: list[Any] = [slice(None)] * 3
    selector[axis] = value
    return selector


def quantise(values: np.ndarray, bins: int = QUANTISATION_BINS) -> np.ndarray:
    """Fixed-bin quantisation of the shell intensities onto ``1..bins``."""
    if values.size == 0:
        return np.zeros(0, dtype=np.int64)
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-9:
        return np.ones(values.shape, dtype=np.int64)
    scaled = (values - low) / (high - low)
    return np.clip(np.floor(scaled * bins).astype(np.int64) + 1, 1, bins)


def first_order(values: np.ndarray, bins: int = QUANTISATION_BINS) -> np.ndarray:
    """First-order statistics over the shell intensities."""
    if values.size == 0:
        return np.zeros(len(FIRST_ORDER_NAMES), dtype=np.float64)
    mean = float(values.mean())
    std = float(values.std())
    centred = values - mean
    variance = float((centred**2).mean())
    third = float((centred**3).mean())
    fourth = float((centred**4).mean())
    skewness = third / (variance**1.5) if variance > 1e-12 else 0.0
    kurtosis = fourth / (variance**2) - 3.0 if variance > 1e-12 else 0.0
    p10, p25, p50, p75, p90 = np.percentile(values, [10, 25, 50, 75, 90])
    histogram = np.bincount(quantise(values, bins), minlength=bins + 2)[1 : bins + 1].astype(np.float64)
    probabilities = histogram / max(histogram.sum(), 1.0)
    positive = probabilities[probabilities > 0]
    entropy = float(-(positive * np.log2(positive)).sum())
    uniformity = float((probabilities**2).sum())
    mad = float(np.abs(centred).mean())
    robust_mad = float(np.abs(values - p50).mean())
    return np.array(
        [
            mean,
            variance,
            skewness,
            kurtosis,
            float(p50),
            float(values.min()),
            float(p10),
            float(p90),
            float(values.max() - values.min()),
            float(p75 - p25),
            mad,
            robust_mad,
            float((values**2).sum()),
            float(np.sqrt((values**2).mean())),
            entropy,
            uniformity,
            float(std / mean) if abs(mean) > 1e-9 else 0.0,
            float(np.median(np.abs(centred))),
        ],
        dtype=np.float64,
    )


def _surface_area(mask: np.ndarray, spacing_mm: tuple[float, float, float]) -> float:
    """Voxel-face surface area, counting the exposed area of each boundary face."""
    area = 0.0
    for axis, _step in enumerate(spacing_mm):
        other = [spacing_mm[index] for index in range(3) if index != axis]
        face = other[0] * other[1]
        lower = mask & ~np.roll(mask, 1, axis=axis)
        upper = mask & ~np.roll(mask, -1, axis=axis)
        selector = [slice(None)] * 3
        selector[axis] = slice(0, 1)
        upper[tuple(selector)] = False
        area += face * float(lower.sum() + upper.sum())
    return area


def shape_features(mask: np.ndarray, spacing_mm: tuple[float, float, float]) -> np.ndarray:
    """Geometry of the descriptor region in millimetres."""
    count = float(mask.sum())
    if count <= 0:
        return np.zeros(len(SHAPE_NAMES), dtype=np.float64)
    voxel_volume = float(np.prod(spacing_mm))
    volume = count * voxel_volume
    surface = _surface_area(mask, spacing_mm)
    sphericity = (np.pi ** (1.0 / 3.0) * (6.0 * volume) ** (2.0 / 3.0) / surface) if surface > 1e-9 else 0.0
    coords = np.argwhere(mask)
    physical = coords.astype(np.float64) * np.asarray(spacing_mm, dtype=np.float64)
    centred = physical - physical.mean(axis=0)
    covariance = centred.T @ centred / max(len(physical) - 1, 1)
    eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    axes = 4.0 * np.sqrt(eigenvalues)
    max_diameter = float(np.max(physical.max(axis=0) - physical.min(axis=0)))
    elongation = float(np.sqrt(eigenvalues[1] / eigenvalues[0])) if eigenvalues[0] > 1e-12 else 0.0
    flatness = float(np.sqrt(eigenvalues[2] / eigenvalues[0])) if eigenvalues[0] > 1e-12 else 0.0
    return np.array(
        [
            volume,
            surface,
            surface / volume if volume > 1e-9 else 0.0,
            sphericity,
            max_diameter,
            float(axes[0]),
            float(axes[2]),
            elongation,
            flatness,
        ],
        dtype=np.float64,
    )


def _glcm_matrices(quant: np.ndarray, mask: np.ndarray, levels: int) -> np.ndarray:
    """Symmetric, normalised co-occurrence matrices averaged over the 13 directions."""
    matrices = np.zeros((len(_DIRECTIONS), levels + 1, levels + 1), dtype=np.float64)
    for index, direction in enumerate(_DIRECTIONS):
        shifted = np.roll(quant, shift=tuple(-component for component in direction), axis=(0, 1, 2))
        shifted_mask = np.roll(mask, shift=tuple(-component for component in direction), axis=(0, 1, 2))
        for axis, component in enumerate(direction):
            if component == 0:
                continue
            edge = _axis_selector(axis, slice(0, 1) if component > 0 else slice(-1, None))
            shifted[tuple(edge)] = 0
            shifted_mask[tuple(edge)] = False
        pairs = mask & shifted_mask
        if not pairs.any():
            continue
        left = quant[pairs]
        right = shifted[pairs]
        counts = np.zeros((levels + 1, levels + 1), dtype=np.float64)
        np.add.at(counts, (left, right), 1.0)
        counts += counts.T
        total = counts.sum()
        if total > 0:
            matrices[index] = counts / total
    return matrices


def glcm_features(quant: np.ndarray, mask: np.ndarray, levels: int = QUANTISATION_BINS) -> np.ndarray:
    """Haralick features averaged over the 13 three-dimensional directions."""
    matrices: np.ndarray = _glcm_matrices(quant, mask, levels)
    usable: np.ndarray = matrices[matrices.sum(axis=(1, 2)) > 0]
    if usable.shape[0] == 0:
        return np.zeros(len(GLCM_NAMES), dtype=np.float64)
    indices = np.arange(levels + 1, dtype=np.float64)
    row, column = np.meshgrid(indices, indices, indexing="ij")
    difference = np.abs(row - column)
    out = np.zeros((usable.shape[0], len(GLCM_NAMES)), dtype=np.float64)
    for position, matrix in enumerate(usable):
        px = matrix.sum(axis=1)
        py = matrix.sum(axis=0)
        ux = float((indices * px).sum())
        uy = float((indices * py).sum())
        sx = float(np.sqrt(((indices - ux) ** 2 * px).sum()))
        sy = float(np.sqrt(((indices - uy) ** 2 * py).sum()))
        correlation = float(((row - ux) * (column - uy) * matrix).sum() / (sx * sy)) if sx > 1e-12 and sy > 1e-12 else 0.0
        positive = matrix[matrix > 0]
        out[position] = [
            float((difference**2 * matrix).sum()),
            float((difference * matrix).sum()),
            float((matrix / (1.0 + difference**2)).sum()),
            float(np.sqrt((matrix**2).sum())),
            correlation,
            float((matrix**2).sum()),
            float(-(positive * np.log2(positive)).sum()),
            float(matrix.max()),
            float((matrix / (1.0 + difference)).sum()),
            _maximal_correlation(matrix),
        ]
    averaged: np.ndarray = out.mean(axis=0)
    return averaged


def _maximal_correlation(matrix: np.ndarray) -> float:
    """Second largest eigenvalue of the normalised co-occurrence matrix."""
    px = matrix.sum(axis=1)
    py = matrix.sum(axis=0)
    if px.sum() <= 0 or py.sum() <= 0:
        return 0.0
    root_px = np.sqrt(np.clip(px, 0.0, None))
    root_py = np.sqrt(np.clip(py, 0.0, None))
    denominator = np.outer(root_px, root_py)
    safe = np.where(denominator > 1e-12, denominator, 1.0)
    normalised = matrix / safe
    with np.errstate(invalid="ignore"):
        eigenvalues = np.linalg.eigvals(normalised)
    values = np.sort(np.real(eigenvalues))[::-1]
    return float(values[1]) if values.size > 1 else 0.0


def _run_length_matrix(quant: np.ndarray, mask: np.ndarray, levels: int) -> np.ndarray:
    """Grey-level run-length counts pooled over the 13 directions."""
    max_run = int(max(quant.shape))
    matrix = np.zeros((levels + 1, max_run + 1), dtype=np.float64)
    for direction in _DIRECTIONS:
        axis = next(index for index, component in enumerate(direction) if component != 0)
        step = direction[axis]
        depth = quant.shape[axis]
        run_lengths = np.zeros_like(quant, dtype=np.int64)
        previous_slice: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        order = range(depth) if step > 0 else range(depth - 1, -1, -1)
        for position in order:
            level_slice = np.take(quant, position, axis=axis)
            mask_slice = np.take(mask, position, axis=axis)
            if previous_slice is None:
                runs = np.ones_like(level_slice)
            else:
                same = (level_slice == previous_slice[0]) & mask_slice & previous_slice[1]
                runs = np.where(same, previous_slice[2] + 1, 1)
            previous_slice = (level_slice, mask_slice, runs)
            run_lengths[tuple(_axis_selector(axis, position))] = np.where(mask_slice, runs, 0)
        active = run_lengths > 0
        if not active.any():
            continue
        np.add.at(matrix, (quant[active], run_lengths[active]), 1.0)
    return matrix


def glrlm_features(quant: np.ndarray, mask: np.ndarray, levels: int = QUANTISATION_BINS) -> np.ndarray:
    """Grey-level run-length features over the 13 directions."""
    matrix = _run_length_matrix(quant, mask, levels)
    total = matrix.sum()
    if total <= 0:
        return np.zeros(len(GLRLM_NAMES), dtype=np.float64)
    normalised = matrix / total
    grey = np.arange(levels + 1, dtype=np.float64)
    runs = np.arange(matrix.shape[1], dtype=np.float64)
    grey_marginal = normalised.sum(axis=1)
    run_marginal = normalised.sum(axis=0)
    return np.array(
        [
            float((normalised / np.where(runs[None, :] > 0, runs[None, :] ** 2, 1.0)).sum()),
            float((normalised * runs[None, :] ** 2).sum()),
            float((grey_marginal**2).sum()),
            float((run_marginal**2).sum()),
            float(total),
            float((normalised * np.where(grey[:, None] > 0, grey[:, None], 1.0) ** -2).sum()),
            float((normalised * grey[:, None] ** 2).sum()),
            float(((grey[:, None] - (normalised * grey[:, None]).sum()) ** 2 * normalised).sum()),
        ],
        dtype=np.float64,
    )


def _connected_zones(quant: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Connected zones of equal quantised level inside the mask.

    Zones are labelled with a 26-connected flood over a boolean volume per level,
    then each zone contributes one (level, size) pair.
    """
    from scipy import ndimage

    levels = np.unique(quant[mask]) if mask.any() else np.zeros(0, dtype=np.int64)
    level_pairs: list[np.ndarray] = []
    size_pairs: list[np.ndarray] = []
    for level in levels:
        binary = mask & (quant == level)
        if not binary.any():
            continue
        labelled, count = ndimage.label(binary, structure=np.ones((3, 3, 3)))
        sizes = np.bincount(labelled.ravel())[1:]
        if count == 0:
            continue
        level_pairs.append(np.full(count, int(level), dtype=np.float64))
        size_pairs.append(sizes.astype(np.float64))
    if not level_pairs:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    return np.concatenate(level_pairs), np.concatenate(size_pairs)


def glszm_features(quant: np.ndarray, mask: np.ndarray, levels: int = QUANTISATION_BINS) -> np.ndarray:
    """Grey-level size-zone features."""
    level_values, zone_sizes = _connected_zones(quant, mask)
    if level_values.size == 0:
        return np.zeros(len(GLSZM_NAMES), dtype=np.float64)
    max_size = int(zone_sizes.max())
    matrix = np.zeros((levels + 1, max_size + 1), dtype=np.float64)
    np.add.at(matrix, (level_values.astype(np.int64), zone_sizes.astype(np.int64)), 1.0)
    total = matrix.sum()
    normalised = matrix / total
    grey = np.arange(levels + 1, dtype=np.float64)
    sizes = np.arange(max_size + 1, dtype=np.float64)
    return np.array(
        [
            float((normalised / np.where(sizes[None, :] > 0, sizes[None, :] ** 2, 1.0)).sum()),
            float((normalised * sizes[None, :] ** 2).sum()),
            float((normalised.sum(axis=1) ** 2).sum()),
            float((normalised.sum(axis=0) ** 2).sum()),
            float(total),
            float((normalised * np.where(grey[:, None] > 0, grey[:, None], 1.0) ** -2).sum()),
            float((normalised * grey[:, None] ** 2).sum()),
            float(((sizes[None, :] - (normalised * sizes[None, :]).sum()) ** 2 * normalised).sum()),
        ],
        dtype=np.float64,
    )


def gldm_features(quant: np.ndarray, mask: np.ndarray, levels: int = QUANTISATION_BINS) -> np.ndarray:
    """Grey-level dependence features, with dependence the count of similar neighbours."""
    dependence = np.zeros_like(quant, dtype=np.int64)
    for direction in _DIRECTIONS:
        shifted = np.roll(quant, shift=tuple(-component for component in direction), axis=(0, 1, 2))
        shifted_mask = np.roll(mask, shift=tuple(-component for component in direction), axis=(0, 1, 2))
        for axis, component in enumerate(direction):
            if component == 0:
                continue
            edge = _axis_selector(axis, slice(0, 1) if component > 0 else slice(-1, None))
            shifted[tuple(edge)] = 0
            shifted_mask[tuple(edge)] = False
        dependence += (np.abs(shifted.astype(np.int64) - quant) <= 1).astype(np.int64) * shifted_mask
    if not mask.any():
        return np.zeros(len(GLDM_NAMES), dtype=np.float64)
    effective = np.maximum(dependence, 1) * mask
    matrix = np.zeros((levels + 1, int(effective.max()) + 2), dtype=np.float64)
    np.add.at(matrix, (quant[mask], effective[mask]), 1.0)
    total = matrix.sum()
    normalised = matrix / total
    depend = np.arange(matrix.shape[1], dtype=np.float64)
    positive = normalised[normalised > 0]
    return np.array(
        [
            float((normalised / np.where(depend[None, :] > 0, depend[None, :] ** 2, 1.0)).sum()),
            float((normalised * depend[None, :] ** 2).sum()),
            float((normalised.sum(axis=1) ** 2).sum()),
            float((normalised.sum(axis=0) ** 2).sum()),
            float(-(positive * np.log2(positive)).sum()),
        ],
        dtype=np.float64,
    )


def ngtdm_features(quant: np.ndarray, mask: np.ndarray, levels: int = QUANTISATION_BINS) -> np.ndarray:
    """Neighbouring grey-tone difference features."""
    if not mask.any():
        return np.zeros(len(NGTDM_NAMES), dtype=np.float64)
    neighbour_sum = np.zeros_like(quant, dtype=np.float64)
    neighbour_count = np.zeros_like(quant, dtype=np.float64)
    for direction in _DIRECTIONS:
        shifted = np.roll(quant, shift=tuple(-component for component in direction), axis=(0, 1, 2))
        shifted_mask = np.roll(mask, shift=tuple(-component for component in direction), axis=(0, 1, 2))
        for axis, component in enumerate(direction):
            if component == 0:
                continue
            edge = _axis_selector(axis, slice(0, 1) if component > 0 else slice(-1, None))
            shifted[tuple(edge)] = 0
            shifted_mask[tuple(edge)] = False
        neighbour_sum += shifted * shifted_mask
        neighbour_count += shifted_mask
    average = np.divide(neighbour_sum, np.maximum(neighbour_count, 1.0))
    valid = mask & (neighbour_count > 0)
    if not valid.any():
        return np.zeros(len(NGTDM_NAMES), dtype=np.float64)
    level_values = quant[valid].astype(np.int64)
    differences = np.abs(level_values - average[valid])
    counts = np.bincount(level_values, minlength=levels + 2)[: levels + 1].astype(np.float64)
    s_values = np.zeros(levels + 1, dtype=np.float64)
    np.add.at(s_values, level_values, differences)
    present = counts > 0
    probabilities = counts / counts.sum()
    n_valid = float(valid.sum())
    coarseness = 1.0 / float((probabilities[present] * s_values[present]).sum()) if (probabilities[present] * s_values[present]).sum() > 1e-12 else 0.0
    grey = np.arange(levels + 1, dtype=np.float64)
    contrast = float(((grey[:, None] - grey[None, :]) ** 2 * np.outer(probabilities, probabilities)).sum()) / n_valid if n_valid > 0 else 0.0
    busyness_num = float((probabilities[present] * s_values[present]).sum())
    spread = np.abs(grey[:, None] * probabilities[:, None] - grey[None, :] * probabilities[None, :])
    busyness_den = float((spread * np.outer(probabilities, probabilities)).sum())
    busyness = busyness_num / busyness_den if busyness_den > 1e-12 else 0.0
    complexity = float((np.abs(grey[:, None] - grey[None, :]) * np.outer(probabilities, probabilities)).sum() / n_valid) if n_valid > 0 else 0.0
    strength = float(((2.0 * probabilities[present]) / np.maximum(s_values[present], 1e-9)).sum()) if present.any() else 0.0
    return np.array([coarseness, contrast, busyness, complexity, strength], dtype=np.float64)


def extract_descriptors(
    volume: np.ndarray,
    mask: np.ndarray,
    spacing_mm: tuple[float, float, float],
    families: tuple[str, ...],
    bins: int = QUANTISATION_BINS,
) -> DescriptorSet:
    """Compute the requested descriptor families over one shell region."""
    unknown = [family for family in families if family not in FAMILY_NAMES]
    if unknown:
        raise ValueError(f"unknown descriptor families: {', '.join(unknown)}")
    values = volume[mask]
    quant_values = quantise(values, bins)
    quant_volume = np.zeros_like(volume, dtype=np.int64)
    quant_volume[mask] = quant_values
    blocks: list[np.ndarray] = []
    names: list[str] = []
    for family in families:
        if family == "firstorder":
            blocks.append(first_order(values, bins))
        elif family == "shape":
            blocks.append(shape_features(mask, spacing_mm))
        elif family == "glcm":
            blocks.append(glcm_features(quant_volume, mask, bins))
        elif family == "glrlm":
            blocks.append(glrlm_features(quant_volume, mask, bins))
        elif family == "glszm":
            blocks.append(glszm_features(quant_volume, mask, bins))
        elif family == "gldm":
            blocks.append(gldm_features(quant_volume, mask, bins))
        else:
            blocks.append(ngtdm_features(quant_volume, mask, bins))
        names.extend(FAMILY_NAMES[family])
    stacked = np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.float64)
    stacked = np.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0)
    return DescriptorSet(names=tuple(names), values=stacked.astype(np.float64))


def intraclass_correlation(repeats: np.ndarray) -> float:
    """ICC(2,1) of one descriptor measured across ``repeats`` segmentations.

    ``repeats`` has shape ``(n_segmentations, n_subjects)``. The two-way
    decomposition is used so that both the between-subject spread and the
    segmentation bias enter the denominator.
    """
    if repeats.ndim != 2:
        raise ValueError("repeats must be (n_segmentations, n_subjects)")
    segmentations, subjects = repeats.shape
    if segmentations < 2 or subjects < 2:
        return 0.0
    grand = float(repeats.mean())
    subject_means = repeats.mean(axis=0)
    segmentation_means = repeats.mean(axis=1)
    ss_subject = segmentations * float(((subject_means - grand) ** 2).sum())
    ss_segmentation = subjects * float(((segmentation_means - grand) ** 2).sum())
    ss_total = float(((repeats - grand) ** 2).sum())
    ss_error = ss_total - ss_subject - ss_segmentation
    ms_subject = ss_subject / (subjects - 1)
    ms_segmentation = ss_segmentation / (segmentations - 1)
    ms_error = ss_error / ((subjects - 1) * (segmentations - 1))
    denominator = ms_subject + (segmentations - 1) * ms_error + segmentations * (ms_segmentation - ms_error) / subjects
    if abs(denominator) < 1e-12:
        return 0.0
    return float((ms_subject - ms_error) / denominator)


def stability_select(
    repeats: np.ndarray,
    names: tuple[str, ...],
    threshold: float,
) -> tuple[list[int], np.ndarray]:
    """Keep the descriptors whose ICC across repeated segmentations reaches ``threshold``.

    ``repeats`` has shape ``(n_segmentations, n_subjects, n_descriptors)``.
    """
    if repeats.ndim != 3:
        raise ValueError("repeats must be (n_segmentations, n_subjects, n_descriptors)")
    if repeats.shape[0] < 2 or repeats.shape[1] < 2:
        return list(range(len(names))), np.ones(len(names), dtype=np.float64)
    scores = np.array([intraclass_correlation(repeats[:, :, column]) for column in range(repeats.shape[2])])
    keep = [index for index, score in enumerate(scores) if score >= threshold]
    if not keep:
        keep = [int(np.argmax(scores))]
    return keep, scores


def collapse_redundant(matrix: np.ndarray, names: list[str], threshold: float) -> list[int]:
    """One representative per block of descriptors correlated above ``threshold``.

    Blocks are formed by complete-linkage clustering on ``1 - |Pearson|`` so that a
    representative is not merely uncorrelated with the descriptor that happened to
    come first.
    """
    if matrix.shape[1] <= 1:
        return list(range(matrix.shape[1]))
    correlation = np.corrcoef(matrix, rowvar=False)
    correlation = np.nan_to_num(correlation, nan=0.0)
    distance = 1.0 - np.abs(correlation)
    np.fill_diagonal(distance, 0.0)
    order = list(range(matrix.shape[1]))
    clusters: list[list[int]] = [[index] for index in order]
    while True:
        best: tuple[float, int, int] | None = None
        for first in range(len(clusters)):
            for second in range(first + 1, len(clusters)):
                worst = max(distance[a, b] for a in clusters[first] for b in clusters[second])
                if best is None or worst < best[0]:
                    best = (worst, first, second)
        if best is None or best[0] > 1.0 - threshold:
            break
        _, first, second = best
        clusters[first] = clusters[first] + clusters[second]
        clusters.pop(second)
    representatives: list[int] = []
    for cluster in clusters:
        centrality = [sum(abs(correlation[member, other]) for other in cluster) for member in cluster]
        representatives.append(cluster[int(np.argmax(centrality))])
    representatives.sort(key=lambda index: names[index])
    return representatives


@dataclass
class SiteStandardiser:
    """Per-site z-scoring fitted on the development split only."""

    locations: dict[str, tuple[np.ndarray, np.ndarray]]

    @classmethod
    def fit(cls, features: np.ndarray, sites: list[str]) -> SiteStandardiser:
        locations: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        site_array = np.asarray(sites)
        for site in sorted(set(sites)):
            block = features[site_array == site]
            mean = block.mean(axis=0)
            std = block.std(axis=0)
            std = np.where(std < 1e-8, 1.0, std)
            locations[site] = (mean.astype(np.float64), std.astype(np.float64))
        return cls(locations=locations)

    def transform(self, features: np.ndarray, sites: list[str]) -> np.ndarray:
        """Standardise using the site's development statistics, falling back to the pooled ones."""
        pooled_mean = np.mean([mean for mean, _ in self.locations.values()], axis=0)
        pooled_std = np.mean([std for _, std in self.locations.values()], axis=0)
        out = np.empty_like(features, dtype=np.float64)
        for row, site in enumerate(sites):
            mean, std = self.locations.get(site, (pooled_mean, pooled_std))
            out[row] = (features[row] - mean) / std
        return out


def default_pipeline(config: RadiomicsConfig) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Family order and the full descriptor name order for the configured families."""
    families = tuple(config.families)
    names = tuple(name for family in families for name in FAMILY_NAMES[family])
    return families, names
