"""EMIT-style SWIR radiance/reflectance utilities.

This module provides small helpers that follow the simplified Lambertian
radiative transfer model used elsewhere in :mod:`alchemi`.  They are intended
for Spectral Watcher Imaging Radiometer (SWIR) processing with EMIT-like
instrument characteristics.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

_MIN_TAU = 1e-3
_MIN_COS_SUN = 1e-3
_MIN_DENOM = 1e-9


def _as_float64(array: np.ndarray | float | Iterable[float]) -> np.ndarray:
    """Convert an input to a ``float64`` :class:`numpy.ndarray`."""

    return np.asarray(array, dtype=np.float64)


def reflectance_to_radiance_emit(
    R: np.ndarray,
    wl_nm: np.ndarray,
    E0: np.ndarray,
    cos_sun: float | np.ndarray,
    tau: float | np.ndarray = 1.0,
    L_path: float | np.ndarray = 0.0,
) -> np.ndarray:
    """Convert surface reflectance to at-sensor SWIR radiance for EMIT-like sensors."""

    R_b, wl_b, E0_b, cos_b, tau_b, Lp_b = np.broadcast_arrays(
        _as_float64(R),
        _as_float64(wl_nm),
        _as_float64(E0),
        _as_float64(cos_sun),
        _as_float64(tau),
        _as_float64(L_path),
    )

    # wl_b is unused in the algebra but ensures that the wavelength grid is
    # broadcast-compatible with the spectrum for downstream consistency.
    del wl_b

    radiance = tau_b * (E0_b * cos_b / np.pi) * R_b + Lp_b
    return radiance


def radiance_to_reflectance_emit(
    L: np.ndarray,
    wl_nm: np.ndarray,
    E0: np.ndarray,
    cos_sun: float | np.ndarray,
    tau: float | np.ndarray = 1.0,
    L_path: float | np.ndarray = 0.0,
) -> np.ndarray:
    """Approximate surface reflectance from EMIT-style SWIR radiance measurements."""

    L_b, wl_b, E0_b, cos_b, tau_b, Lp_b = np.broadcast_arrays(
        _as_float64(L),
        _as_float64(wl_nm),
        _as_float64(E0),
        _as_float64(cos_sun),
        _as_float64(tau),
        _as_float64(L_path),
    )

    del wl_b

    tau_b = np.clip(tau_b, _MIN_TAU, None)
    cos_b = np.clip(cos_b, _MIN_COS_SUN, None)
    E0_b = np.clip(E0_b, _MIN_DENOM, None)

    denom = tau_b * (E0_b * cos_b / np.pi)
    denom = np.clip(denom, _MIN_DENOM, None)

    R = (L_b - Lp_b) / denom
    R = np.clip(R, 0.0, 2.0)
    return R


def continuum_removed(
    R: np.ndarray,
    wl_nm: np.ndarray,
) -> np.ndarray:
    """Compute continuum-removed reflectance for EMIT-style SWIR spectra."""

    R_b, wl_b = np.broadcast_arrays(_as_float64(R), _as_float64(wl_nm))

    if wl_b.ndim != 1:
        raise ValueError("wl_nm must be a one-dimensional wavelength grid")

    spectra = R_b.reshape(-1, R_b.shape[-1])
    continuum = np.empty_like(spectra)

    for idx, spec in enumerate(spectra):
        continuum[idx] = _upper_hull_continuum(wl_b, spec)

    continuum = continuum.reshape(R_b.shape)
    continuum = np.clip(continuum, _MIN_DENOM, None)
    removed = R_b / continuum
    return removed


def band_depth(
    R: np.ndarray,
    wl_nm: np.ndarray,
    left_nm: float,
    center_nm: float,
    right_nm: float,
) -> np.ndarray:
    """Compute absorption band depth for EMIT-like spectra."""

    if not (left_nm < center_nm < right_nm):
        raise ValueError("Expected left_nm < center_nm < right_nm for band depth")

    R_b, wl_b = np.broadcast_arrays(_as_float64(R), _as_float64(wl_nm))

    if wl_b.ndim != 1:
        raise ValueError("wl_nm must be a one-dimensional wavelength grid")

    spectra = R_b.reshape(-1, R_b.shape[-1])
    n_spec = spectra.shape[0]

    left_vals = np.empty(n_spec, dtype=np.float64)
    center_vals = np.empty(n_spec, dtype=np.float64)
    right_vals = np.empty(n_spec, dtype=np.float64)

    for idx, spec in enumerate(spectra):
        left_vals[idx] = np.interp(left_nm, wl_b, spec)
        center_vals[idx] = np.interp(center_nm, wl_b, spec)
        right_vals[idx] = np.interp(right_nm, wl_b, spec)

    slope = (right_vals - left_vals) / (right_nm - left_nm)
    continuum_center = left_vals + slope * (center_nm - left_nm)
    continuum_center = np.clip(continuum_center, _MIN_DENOM, None)

    depth = 1.0 - center_vals / continuum_center
    depth = depth.reshape(R_b.shape[:-1])
    return depth


def _upper_hull_continuum(wavelengths: np.ndarray, spectrum: np.ndarray) -> np.ndarray:
    """Construct a simple upper hull continuum for a single spectrum."""

    hull_x: list[float] = []
    hull_y: list[float] = []

    for x, y in zip(wavelengths, spectrum, strict=True):
        hull_x.append(float(x))
        hull_y.append(float(y))
        while len(hull_x) >= 3:
            x0, y0 = hull_x[-3], hull_y[-3]
            x1, y1 = hull_x[-2], hull_y[-2]
            x2, y2 = hull_x[-1], hull_y[-1]
            cross = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
            if cross >= 0.0:
                hull_x.pop(-2)
                hull_y.pop(-2)
            else:
                break

    hull_x_arr = np.asarray(hull_x, dtype=np.float64)
    hull_y_arr = np.asarray(hull_y, dtype=np.float64)

    continuum = np.interp(wavelengths, hull_x_arr, hull_y_arr)
    return continuum

