"""AVIRIS-NG specific SWIR radiance/reflectance helpers."""

from __future__ import annotations

import numpy as np

_BAD_BAND_WINDOWS = ((1340.0, 1440.0), (1800.0, 1950.0))


def avirisng_bad_band_mask(wl_nm: np.ndarray) -> np.ndarray:
    """Return a boolean mask for valid AVIRIS-NG SWIR bands."""

    wavelengths = np.asarray(wl_nm, dtype=float)
    mask = np.ones_like(wavelengths, dtype=bool)

    for low, high in _BAD_BAND_WINDOWS:
        mask &= ~((wavelengths >= low) & (wavelengths <= high))

    return mask


def _resolve_band_mask(wl_nm: np.ndarray, band_mask: np.ndarray | None) -> np.ndarray:
    wavelengths = np.asarray(wl_nm, dtype=float)
    if band_mask is None:
        mask = avirisng_bad_band_mask(wavelengths)
    else:
        mask = np.asarray(band_mask, dtype=bool)
        if mask.shape != wavelengths.shape:
            raise ValueError("band_mask must match wl_nm shape")

    return mask


def reflectance_to_radiance_avirisng(
    R: np.ndarray,
    wl_nm: np.ndarray,
    E0: np.ndarray,
    cos_sun: float | np.ndarray,
    tau: float | np.ndarray = 1.0,
    L_path: float | np.ndarray = 0.0,
    band_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Convert reflectance to radiance for AVIRIS-NG with bad-band handling.

    Uses the single-layer TOA approximation; surface reflectance should be
    sourced from L2A products rather than inferred here.
    """

    mask = _resolve_band_mask(wl_nm, band_mask)

    dtype = np.result_type(R, E0, cos_sun, tau, L_path, np.float32)
    R_arr = np.asarray(R, dtype=dtype)
    E0_arr = np.asarray(E0, dtype=dtype)
    cos_arr = np.asarray(cos_sun, dtype=dtype)
    tau_arr = np.asarray(tau, dtype=dtype)
    L_path_arr = np.asarray(L_path, dtype=dtype)

    if R_arr.shape != mask.shape:
        raise ValueError("Reflectance and wl_nm must share the same shape")

    for name, arr in {
        "E0": E0_arr,
        "cos_sun": cos_arr,
        "tau": tau_arr,
        "L_path": L_path_arr,
    }.items():
        if arr.shape not in (mask.shape, ()):  # Scalars or per-band arrays
            raise ValueError(f"{name} must be scalar or match wl_nm shape")

    scale = tau_arr * (E0_arr * cos_arr / np.pi)
    radiance = scale * R_arr + L_path_arr

    invalid_fill = np.full_like(radiance, np.nan)
    return np.where(mask, radiance, invalid_fill)


def radiance_to_reflectance_avirisng(
    L: np.ndarray,
    wl_nm: np.ndarray,
    E0: np.ndarray,
    cos_sun: float | np.ndarray,
    tau: float | np.ndarray = 1.0,
    L_path: float | np.ndarray = 0.0,
    band_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Convert radiance to reflectance for AVIRIS-NG with bad-band handling.

    The inversion mirrors :func:`reflectance_to_radiance_avirisng` and remains a
    TOA-level approximation rather than a full atmospheric correction.
    """

    mask = _resolve_band_mask(wl_nm, band_mask)

    dtype = np.result_type(L, E0, cos_sun, tau, L_path, np.float32)
    L_arr = np.asarray(L, dtype=dtype)
    E0_arr = np.asarray(E0, dtype=dtype)
    cos_arr = np.asarray(cos_sun, dtype=dtype)
    tau_arr = np.asarray(tau, dtype=dtype)
    L_path_arr = np.asarray(L_path, dtype=dtype)

    if L_arr.shape != mask.shape:
        raise ValueError("Radiance and wl_nm must share the same shape")

    for name, arr in {
        "E0": E0_arr,
        "cos_sun": cos_arr,
        "tau": tau_arr,
        "L_path": L_path_arr,
    }.items():
        if arr.shape not in (mask.shape, ()):  # Scalars or per-band arrays
            raise ValueError(f"{name} must be scalar or match wl_nm shape")

    denom = tau_arr * (E0_arr * cos_arr / np.pi)
    denom = np.clip(denom, 1e-12, None)

    with np.errstate(divide="ignore", invalid="ignore"):
        reflectance = (L_arr - L_path_arr) / denom

    invalid_fill = np.full_like(reflectance, np.nan)
    return np.where(mask, reflectance, invalid_fill)
