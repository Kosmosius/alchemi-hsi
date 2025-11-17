from __future__ import annotations

import numpy as np


def _to_dtype_array(value: np.ndarray | float, dtype: np.dtype) -> np.ndarray:
    return np.asarray(value, dtype=dtype)


def interpolate_irradiance_to_bands(
    wl_band_nm: np.ndarray,
    wl_E0_nm: np.ndarray,
    E0_nm: np.ndarray,
    *,
    fill_value: float | None = None,
) -> np.ndarray:
    """Interpolate a high-resolution solar irradiance spectrum onto EnMAP's band grid."""
    wl_band = np.asarray(wl_band_nm, dtype=np.float64)
    wl_E0 = np.asarray(wl_E0_nm, dtype=np.float64)
    E0 = np.asarray(E0_nm, dtype=np.float64)

    if wl_band.ndim != 1 or wl_E0.ndim != 1 or E0.ndim != 1:
        raise ValueError("All wavelength and irradiance arrays must be one-dimensional.")
    if wl_E0.shape[0] != E0.shape[0]:
        raise ValueError("wl_E0_nm and E0_nm must have matching shapes.")

    if fill_value is None:
        left = None
        right = None
    else:
        fill = float(fill_value)
        left = fill
        right = fill

    return np.interp(wl_band, wl_E0, E0, left=left, right=right)


def _validate_band_inputs(
    spectrum: np.ndarray,
    wl_nm: np.ndarray,
    E0: np.ndarray,
    band_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    if spectrum.shape[-1] != wl_nm.shape[0]:
        raise ValueError("Last axis of spectrum must match wavelength dimension.")

    if E0.shape[-1] != wl_nm.shape[0]:
        raise ValueError("E0 must share the wavelength dimension.")

    if band_mask is not None:
        mask = np.asarray(band_mask, dtype=bool)
        if mask.shape != (wl_nm.shape[0],):
            raise ValueError("band_mask must be one-dimensional with length equal to bands.")
    else:
        mask = None

    return spectrum, wl_nm, E0, mask


def _ensure_dtype(*arrays: np.ndarray | float) -> np.dtype:
    return np.result_type(*arrays, np.float32)


def reflectance_to_radiance_enmap(
    R: np.ndarray,
    wl_nm: np.ndarray,
    E0: np.ndarray,
    cos_sun: float | np.ndarray,
    tau: float | np.ndarray = 1.0,
    L_path: float | np.ndarray = 0.0,
    band_mask: np.ndarray | None = None,
) -> np.ndarray:
    """EnMAP-specific reflectance→radiance using the same RT model as EMIT."""
    wl = np.asarray(wl_nm)
    E0_arr = np.asarray(E0)
    R_arr = np.asarray(R)

    _validate_band_inputs(R_arr, wl, E0_arr, band_mask)

    dtype = _ensure_dtype(R_arr, wl, E0_arr, cos_sun, tau, L_path)

    wl = _to_dtype_array(wl, dtype)
    E0_arr = _to_dtype_array(E0_arr, dtype)
    R_arr = _to_dtype_array(R_arr, dtype)
    cos_arr = _to_dtype_array(cos_sun, dtype)
    tau_arr = _to_dtype_array(tau, dtype)
    L_path_arr = _to_dtype_array(L_path, dtype)

    pi = dtype.type(np.pi)
    scale = tau_arr * (E0_arr * cos_arr / pi)
    L = scale * R_arr + L_path_arr

    if band_mask is not None:
        mask = np.asarray(band_mask, dtype=bool)
        invalid = ~mask
        if invalid.any():
            L = np.array(L, copy=True)
            L[..., invalid] = np.nan

    return L


def radiance_to_reflectance_enmap(
    L: np.ndarray,
    wl_nm: np.ndarray,
    E0: np.ndarray,
    cos_sun: float | np.ndarray,
    tau: float | np.ndarray = 1.0,
    L_path: float | np.ndarray = 0.0,
    band_mask: np.ndarray | None = None,
) -> np.ndarray:
    """EnMAP-specific radiance→reflectance inverse."""
    wl = np.asarray(wl_nm)
    E0_arr = np.asarray(E0)
    L_arr = np.asarray(L)

    _validate_band_inputs(L_arr, wl, E0_arr, band_mask)

    dtype = _ensure_dtype(L_arr, wl, E0_arr, cos_sun, tau, L_path)

    wl = _to_dtype_array(wl, dtype)
    E0_arr = _to_dtype_array(E0_arr, dtype)
    L_arr = _to_dtype_array(L_arr, dtype)
    cos_arr = _to_dtype_array(cos_sun, dtype)
    tau_arr = _to_dtype_array(tau, dtype)
    L_path_arr = _to_dtype_array(L_path, dtype)

    pi = dtype.type(np.pi)
    denom = tau_arr * (E0_arr * cos_arr / pi)
    numerator = L_arr - L_path_arr
    with np.errstate(divide="ignore", invalid="ignore"):
        R = numerator / denom
    R = np.where(denom == 0, np.nan, R)

    if band_mask is not None:
        mask = np.asarray(band_mask, dtype=bool)
        invalid = ~mask
        if invalid.any():
            R = np.array(R, copy=True)
            R[..., invalid] = np.nan

    return R


__all__ = [
    "interpolate_irradiance_to_bands",
    "radiance_to_reflectance_enmap",
    "reflectance_to_radiance_enmap",
]
