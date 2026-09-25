"""Image acquisition and preprocessing.

The pipeline follows the manuscript's preprocessing contract: a portal-venous
phase volume is resampled to 1 x 1 x 1 mm isotropic spacing, windowed to the
abdominal range, corrected for the low-frequency intensity inhomogeneity that
surface-coil and reconstruction differences produce, and standardised within the
examination. Augmentation is limited to changes that do not alter staging-relevant
anatomy: a bounded affine perturbation and a bounded intensity jitter.

Because the clinical volumes are not redistributable, ``synthesise_volume`` builds
a volume from a record's imaging latent on demand. The construction is separable:
a coarse grid is projected from the latent and smoothly upsampled, so the low
frequency content of the volume carries the latent and a patch-based encoder can
recover it.

Ref: Methods Sec. 4.3 (image acquisition and preprocessing); Sec. 4.7
(augmentations).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from ..utils.config import ImagingConfig

COARSE_SHAPE = (6, 6, 4)
AIR_HU = -1000.0


@dataclass(frozen=True)
class VolumeGeometry:
    """Spacing and grid of one cached volume."""

    spacing_mm: tuple[float, float, float]
    shape: tuple[int, int, int]

    @property
    def extent_mm(self) -> tuple[float, float, float]:
        return tuple(count * step for count, step in zip(self.shape, self.spacing_mm))  # type: ignore[return-value]


def resample_isotropic(volume: np.ndarray, spacing_mm: tuple[float, float, float], target_mm: tuple[float, float, float]) -> np.ndarray:
    """Resample to isotropic spacing with trilinear interpolation."""
    if len(spacing_mm) != 3:
        raise ValueError("spacing must have three components")
    factors = tuple(source / target for source, target in zip(spacing_mm, target_mm))
    if all(abs(factor - 1.0) < 1e-6 for factor in factors):
        return volume.astype(np.float32, copy=False)
    order = 1
    resampled: np.ndarray = ndimage.zoom(volume.astype(np.float32), factors, order=order, mode="nearest")
    return resampled.astype(np.float32)


def apply_hu_window(volume: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    """Clip to the abdominal Hounsfield window and rescale to [0, 1]."""
    low, high = float(window[0]), float(window[1])
    clipped: np.ndarray = np.clip(volume, low, high)
    return ((clipped - low) / (high - low)).astype(np.float32)


def estimate_bias_field(volume: np.ndarray, sigma_voxels: float) -> np.ndarray:
    """Low-frequency multiplicative inhomogeneity estimated from the volume itself.

    The field is the heavily smoothed volume normalised to unit mean, which is the
    classic retrospective correction for smooth acquisition shading.
    """
    smoothed: np.ndarray = ndimage.gaussian_filter(volume.astype(np.float32), sigma=sigma_voxels, mode="nearest")
    mean = float(smoothed.mean())
    if not np.isfinite(mean) or abs(mean) < 1e-6:
        return np.ones_like(volume, dtype=np.float32)
    return (smoothed / mean).astype(np.float32)


def correct_bias_field(volume: np.ndarray, spacing_mm: tuple[float, float, float], sigma_mm: float) -> np.ndarray:
    """Divide out the estimated shading field, guarding against near-zero divisors."""
    sigma_voxels = float(np.mean([max(sigma_mm / max(step, 1e-3), 1.0) for step in spacing_mm]))
    field: np.ndarray = np.clip(estimate_bias_field(volume, sigma_voxels), 0.4, 2.5)
    corrected: np.ndarray = volume.astype(np.float32) / field
    return corrected.astype(np.float32)


def standardise_intensity(volume: np.ndarray, method: str = "zscore") -> np.ndarray:
    """Standardise intensities within the examination."""
    if method == "zscore":
        mean = float(volume.mean())
        std = float(volume.std())
        return ((volume - mean) / (std if std > 1e-6 else 1.0)).astype(np.float32)
    if method == "minmax":
        low, high = float(volume.min()), float(volume.max())
        span = high - low if high - low > 1e-6 else 1.0
        return ((volume - low) / span).astype(np.float32)
    raise ValueError(f"unknown intensity standardisation: {method}")


def preprocess_volume(volume: np.ndarray, spacing_mm: tuple[float, float, float], config: ImagingConfig) -> np.ndarray:
    """Full preprocessing chain for one examination."""
    isotropic = resample_isotropic(volume, spacing_mm, config.spacing_mm)
    corrected = correct_bias_field(isotropic, config.spacing_mm, config.bias_field_sigma_mm)
    windowed = apply_hu_window(corrected, config.hu_window)
    return standardise_intensity(windowed, config.intensity_standardisation)


def _projection_matrix(latent_dim: int, coarse_voxels: int, seed: int) -> np.ndarray:
    """Fixed random projection from the latent onto the coarse grid."""
    rng = np.random.default_rng(seed)
    drawn: np.ndarray = rng.normal(size=(coarse_voxels, latent_dim)) / np.sqrt(latent_dim)
    return drawn.astype(np.float32)


def synthesise_volume(
    latent: np.ndarray,
    grid: tuple[int, int, int],
    seed: int,
    site_offset: float = 0.0,
    site_texture: float = 0.0,
    noise: float = 0.05,
) -> np.ndarray:
    """Build a portal-venous-like volume carrying ``latent`` in its coarse content.

    Acquisition differences between sites enter as a smooth additive offset and a
    vendor-specific texture amplitude, which is what makes the distribution-shift
    analysis well posed.
    """
    depth, height, width = grid
    coarse_voxels = int(np.prod(COARSE_SHAPE))
    projection = _projection_matrix(int(latent.shape[0]), coarse_voxels, seed)
    coarse = (projection @ latent.astype(np.float32)).reshape(COARSE_SHAPE)
    factors = (depth / COARSE_SHAPE[0], height / COARSE_SHAPE[1], width / COARSE_SHAPE[2])
    zoomed: np.ndarray = ndimage.zoom(coarse, factors, order=3, mode="nearest")
    body = zoomed[:depth, :height, :width]
    if body.shape != grid:
        padded = np.zeros(grid, dtype=np.float32)
        padded[: body.shape[0], : body.shape[1], : body.shape[2]] = body
        body = padded
    rng = np.random.default_rng(seed + 17)
    texture: np.ndarray = rng.normal(size=grid).astype(np.float32)
    texture = np.asarray(ndimage.gaussian_filter(texture, sigma=2.0, mode="nearest"), dtype=np.float32)
    field = body + site_offset + site_texture * texture + noise * rng.normal(size=grid)
    anatomy = 100.0 * np.tanh(field.astype(np.float32)) - 60.0
    return anatomy.astype(np.float32)


def synthesise_patient_volume(latent: np.ndarray, grid: tuple[int, int, int], config: ImagingConfig, site_index: int, seed: int) -> np.ndarray:
    """Synthesise and preprocess one examination's volume, fitted to the model grid."""
    vendor_offset = 12.0 * np.sin(0.7 * site_index)
    raw = synthesise_volume(latent, grid, seed, site_offset=vendor_offset, site_texture=0.25 * site_index, noise=0.06)
    return fit_to_grid(preprocess_volume(raw, (1.2, 0.9, 0.9), config), grid)


def augment_volume(
    volume: np.ndarray,
    rng: np.random.Generator,
    rotate_degrees: float,
    scale_fraction: float,
    jitter_hu: float,
    intensity_scale: float = 240.0,
) -> np.ndarray:
    """Apply the manuscript's augmentation set to one preprocessed volume.

    Only anatomy-preserving perturbations are used: a bounded rotation about the
    cranio-caudal axis, a bounded isotropic rescaling, and an intensity jitter.
    """
    angle = float(rng.uniform(-rotate_degrees, rotate_degrees))
    rotated: np.ndarray = ndimage.rotate(volume, angle, axes=(1, 2), reshape=False, order=1, mode="nearest")
    factor = float(rng.uniform(1.0 - scale_fraction, 1.0 + scale_fraction))
    scaled: np.ndarray = ndimage.zoom(rotated, factor, order=1, mode="nearest")
    if scaled.shape != volume.shape:
        cropped = np.zeros_like(volume)
        slices = tuple(slice(0, min(scaled.shape[axis], volume.shape[axis])) for axis in range(3))
        cropped[slices] = scaled[slices]
        scaled = cropped
    jitter = float(rng.uniform(-jitter_hu, jitter_hu))
    return (scaled + jitter / intensity_scale).astype(np.float32)


def fit_to_grid(volume: np.ndarray, grid: tuple[int, int, int]) -> np.ndarray:
    """Centre-crop or centre-pad a volume to exactly ``grid``.

    The manuscript fixes the grid the model sees, so the resampled volume is fitted
    to it rather than inheriting whatever shape the source spacing produced.
    """
    fitted = np.zeros(grid, dtype=np.float32)
    source_slices: list[slice] = []
    target_slices: list[slice] = []
    for axis, target in enumerate(grid):
        current = volume.shape[axis]
        if current >= target:
            start = (current - target) // 2
            source_slices.append(slice(start, start + target))
            target_slices.append(slice(0, target))
        else:
            start = (target - current) // 2
            source_slices.append(slice(0, current))
            target_slices.append(slice(start, start + current))
    fitted[tuple(target_slices)] = volume[tuple(source_slices)]
    return fitted


def patchify(volume: np.ndarray, patch_size: int) -> np.ndarray:
    """Split a volume into non-overlapping cubic patches, flattened to the last axis.

    Returns ``(D/P, H/P, W/P, P^3)``; a trailing partial patch along an axis is
    dropped so the token count is exact rather than padded.
    """
    depth, height, width = volume.shape
    counts = (depth // patch_size, height // patch_size, width // patch_size)
    if min(counts) == 0:
        raise ValueError(f"volume {volume.shape} is smaller than patch size {patch_size}")
    trimmed = volume[: counts[0] * patch_size, : counts[1] * patch_size, : counts[2] * patch_size]
    reshaped = trimmed.reshape(counts[0], patch_size, counts[1], patch_size, counts[2], patch_size)
    tokens = reshaped.transpose(0, 2, 4, 1, 3, 5).reshape(counts[0] * counts[1] * counts[2], patch_size**3)
    return tokens.astype(np.float32)


def token_grid(grid: tuple[int, int, int], patch_size: int) -> tuple[int, int, int]:
    """Token counts along each axis for a grid and patch size."""
    return tuple(int(length // patch_size) for length in grid)  # type: ignore[return-value]
