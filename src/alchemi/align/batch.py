from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

__all__ = ["build_emit_pairs", "build_enmap_pairs"]


def _as_2d(values: np.ndarray | Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        return arr[None, :]
    if arr.ndim != 2:
        msg = "Lab spectra must be a 1-D or 2-D array"
        raise ValueError(msg)
    return arr


def _ensure_rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    if isinstance(rng, (int, np.integer)):
        return np.random.default_rng(int(rng))
    if rng is None:
        return np.random.default_rng()
    msg = "rng must be a numpy.random.Generator, integer seed, or None"
    raise TypeError(msg)


def _project_lab(
    lab_nm: np.ndarray | Sequence[float],
    lab_reflectance: np.ndarray | Sequence[Sequence[float]],
    sensor: str,
    cache_dir: str | Path | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lab_nm_arr = np.asarray(lab_nm, dtype=np.float64)
    if lab_nm_arr.ndim != 1 or not np.all(np.diff(lab_nm_arr) > 0):
        msg = "lab_nm must be a strictly increasing 1-D array"
        raise ValueError(msg)
    lab_values = _as_2d(lab_reflectance)
    if lab_values.shape[1] != lab_nm_arr.shape[0]:
        msg = "lab_reflectance length must match lab_nm"
        raise ValueError(msg)

    if sensor.lower() == "emit":
        from ..srf.batch_convolve import batch_convolve_lab_to_sensor as _convolve
        from ..srf.emit import emit_srf_matrix

        srf = emit_srf_matrix(lab_nm_arr)
        centers = srf.centers_nm
        convolve = _convolve
    elif sensor.lower() == "enmap":
        from ..srf.batch_convolve import batch_convolve_lab_to_sensor as _convolve
        from ..srf.enmap import enmap_srf_matrix

        if cache_dir is None:
            srf = enmap_srf_matrix()
        else:
            srf = enmap_srf_matrix(cache_dir=cache_dir)
        centers = srf.centers_nm
        convolve = _convolve
    else:  # pragma: no cover - defensive
        msg = f"Unsupported sensor '{sensor}'"
        raise ValueError(msg)

    projected = convolve(lab_nm_arr, lab_values, srf)
    mask = np.ones_like(centers, dtype=bool)
    return centers, projected, mask


def build_emit_pairs(
    lab_nm: np.ndarray | Sequence[float],
    lab_reflectance: np.ndarray | Sequence[Sequence[float]],
    *,
    noise_level_rel: float = 0.0,
    rng: np.random.Generator | int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project lab spectra to the EMIT grid and optionally inject noise."""

    centers, projected, mask = _project_lab(lab_nm, lab_reflectance, "emit", None)
    generator = _ensure_rng(rng)
    noise_scale = float(noise_level_rel)
    if noise_scale < 0:
        raise ValueError("noise_level_rel must be non-negative")

    if noise_scale == 0.0:
        sensor = projected.copy()
    else:
        sigma = noise_scale * np.maximum(np.abs(projected), 1e-8)
        noise = generator.normal(loc=0.0, scale=sigma, size=projected.shape)
        sensor = projected + noise
    return centers, projected, sensor, mask


def build_enmap_pairs(
    lab_nm: np.ndarray | Sequence[float],
    lab_reflectance: np.ndarray | Sequence[Sequence[float]],
    *,
    cache_dir: str | Path | None = "data/srf",
    noise_level_rel_vnir: float = 0.0,
    noise_level_rel_swir: float = 0.0,
    rng: np.random.Generator | int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project lab spectra to the merged EnMAP VNIR+SWIR grid with optional noise.

    Parameters
    ----------
    lab_nm:
        Strictly increasing wavelength grid for the laboratory spectra.
    lab_reflectance:
        Lab reflectance spectra sampled on ``lab_nm``.
    cache_dir:
        Directory used for caching the synthesized EnMAP SRF JSON resource.
        Provide ``None`` to use the default location.
    noise_level_rel_vnir, noise_level_rel_swir:
        Relative noise levels (1σ) applied to the projected spectra for the
        VNIR (≤999 nm) and SWIR (>999 nm) spectrometers, respectively.
    rng:
        Optional ``numpy.random.Generator`` or seed controlling the noise
        sampling. When omitted a new generator is instantiated.

    Returns
    -------
    wavelengths_nm, lab_convolved, sensor_noisy, band_mask
    """

    if noise_level_rel_vnir < 0 or noise_level_rel_swir < 0:
        raise ValueError("noise levels must be non-negative")

    centers, projected, mask = _project_lab(lab_nm, lab_reflectance, "enmap", cache_dir)
    generator = _ensure_rng(rng)

    rel_levels = np.where(centers <= 999.0, float(noise_level_rel_vnir), float(noise_level_rel_swir))
    if np.any(rel_levels > 0.0):
        sigma = rel_levels[None, :] * np.maximum(np.abs(projected), 1e-8)
        noise = generator.normal(loc=0.0, scale=sigma, size=projected.shape)
        sensor = projected + noise
    else:
        sensor = projected.copy()
    return centers, projected, sensor, mask
