from __future__ import annotations

from typing import Any

import numpy as np

from ..srf import SRFRegistry, batch_convolve_lab_to_sensor

__all__ = ["build_avirisng_pairs"]


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

    AVIRIS-NG provides 380–2510 nm coverage across hundreds of ≈5 nm full-width
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
        When omitted, the default registry rooted at ``data/srf`` is used.
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

    registry = srf_registry or SRFRegistry()
    srf = registry.get("avirisng")
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
