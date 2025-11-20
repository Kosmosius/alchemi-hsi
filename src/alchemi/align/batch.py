from __future__ import annotations

# mypy: ignore-errors
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..srf import SRFRegistry, batch_convolve_lab_to_sensor
from ..srf.avirisng import avirisng_srf_matrix

__all__ = ["build_avirisng_pairs", "build_emit_pairs", "build_enmap_pairs"]


def _as_2d(values: np.ndarray | Sequence[float] | Sequence[Sequence[float]]) -> np.ndarray:
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

    projected = np.asarray(convolve(lab_nm_arr, lab_values, srf), dtype=np.float64)
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
        Relative noise levels (1-sigma) applied to the projected spectra for the
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

    rel_levels = np.where(
        centers <= 999.0,
        float(noise_level_rel_vnir),
        float(noise_level_rel_swir),
    )
    if np.any(rel_levels > 0.0):
        sigma = rel_levels[None, :] * np.maximum(np.abs(projected), 1e-8)
        noise = generator.normal(loc=0.0, scale=sigma, size=projected.shape)
        sensor = projected + noise
    else:
        sensor = projected.copy()
    return centers, projected, sensor, mask


# --- AVIRIS-NG helpers -----------------------------------------------------


def _validate_wavelengths(lab_nm: np.ndarray) -> np.ndarray:
    nm = np.asarray(lab_nm, dtype=np.float64)
    if nm.ndim != 1:
        raise ValueError("lab_nm must be a 1-D array")
    if nm.size < 2:
        raise ValueError("lab_nm must contain at least two samples")
    if np.any(np.diff(nm) <= 0.0):
        raise ValueError("lab_nm must be strictly increasing")
    return nm


def _validate_lab_values(lab_values: np.ndarray, expected: int) -> np.ndarray:
    values = np.asarray(lab_values, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2:
        raise ValueError("lab_values must be a 2-D array or broadcastable to 2-D")
    if values.shape[1] != expected:
        raise ValueError("lab_values second dimension must match lab_nm length")
    return values


def _resolve_noise(noise: float | np.ndarray | None, band_count: int) -> np.ndarray | None:
    if noise is None:
        return None
    coeff = np.asarray(noise, dtype=np.float64)
    if coeff.ndim == 0:
        coeff = np.full((band_count,), float(coeff), dtype=np.float64)
    elif coeff.shape != (band_count,):
        raise ValueError("noise must be scalar or shape (band_count,)")
    if np.any(coeff < 0.0):
        raise ValueError("noise coefficients must be non-negative")
    return coeff


def _resolve_rng(seed: int | np.random.Generator | None) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def build_avirisng_pairs(
    lab_nm: np.ndarray,
    lab_values: np.ndarray,
    *,
    srf_registry: SRFRegistry | None = None,
    noise: float | np.ndarray | None = None,
    seed: int | np.random.Generator | None = None,
) -> dict[str, Any]:
    """Project lab spectra onto the AVIRIS-NG band grid and add optional noise.

    AVIRIS-NG provides 380-2510 nm coverage across hundreds of ≈5 nm full-width
    half-maximum bands (`JPL`_). The sensor response functions (SRFs) encode the
    precise bandpass for each channel; this helper projects laboratory spectra onto
    that grid so they can be compared with airborne observations.

    .. _JPL: https://avirisng.jpl.nasa.gov

    Parameters
    ----------
    lab_nm:
        Wavelength grid for the laboratory spectra (nanometres). The values must be
        strictly increasing.
    lab_values:
        Batch of laboratory reflectance spectra sampled on ``lab_nm``. The array is
        interpreted as ``[batch, wavelength]``. A single spectrum may be provided as a
        1-D array.
    srf_registry:
        Optional SRF registry used to look up the AVIRIS-NG sensor response function.
        When omitted, a built-in procedural SRF generator is used so the helper works
        out of the box without requiring serialized SRF assets.
    noise:
        Optional per-band relative noise level. Scalars apply the same coefficient to
        every band; arrays must match the AVIRIS-NG band count. Noise is sampled from a
        zero-mean normal distribution with ``sigma = noise * abs(values)``.
    seed:
        Random seed or generator used when sampling noise.

    Returns
    -------
    dict
        A dictionary containing the projected wavelengths, convolved spectra, and the
        AVIRIS-NG bad-band mask. The keys are ``"wavelengths_nm"``, ``"lab_values"``,
        and ``"band_mask"``.
    """

    nm = _validate_wavelengths(lab_nm)
    values = _validate_lab_values(lab_values, nm.shape[0])

    if srf_registry is None:
        srf = avirisng_srf_matrix()
    else:
        srf = srf_registry.get("avirisng")
    centers = np.asarray(srf.centers_nm, dtype=np.float64)

    projected = batch_convolve_lab_to_sensor(nm, values, srf)

    band_mask = getattr(srf, "bad_band_mask", None)
    if band_mask is None:
        mask = np.ones_like(centers, dtype=bool)
    else:
        mask = np.asarray(band_mask, dtype=bool)
        if mask.shape != centers.shape:
            raise ValueError("SRF bad band mask must match band centers shape")

    coeff = _resolve_noise(noise, centers.shape[0])
    if coeff is not None:
        rng = _resolve_rng(seed)
        coeff = coeff.reshape(1, -1)
        sigma = coeff * np.abs(projected)
        noise_samples = rng.normal(loc=0.0, scale=sigma, size=projected.shape)
        projected = projected + noise_samples

    return {
        "wavelengths_nm": centers,
        "lab_values": projected,
        "band_mask": mask,
    }
