"""EMIT-style SWIR radiance/reflectance utilities.

This module provides small helpers that follow the simplified Lambertian
radiative transfer model used elsewhere in :mod:`alchemi`.  They are intended
for Spectral Watcher Imaging Radiometer (SWIR) processing with EMIT-like
instrument characteristics.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from alchemi.physics.continuum import BandDefinition, build_continuum, compute_band_metrics
from alchemi.types import Spectrum, WavelengthGrid

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

    wl = _as_float64(wl_nm)
    R_b = _as_float64(R)

    if wl.ndim != 1:
        raise ValueError("wl_nm must be a one-dimensional wavelength grid")
    if R_b.shape[-1] != wl.size:
        msg = "Last dimension of R must match wavelength grid length"
        raise ValueError(msg)

    spectrum = Spectrum.from_reflectance(WavelengthGrid(wl), R_b)
    continuum = build_continuum(spectrum, method="convex_hull")
    continuum = np.clip(continuum, _MIN_DENOM, None)
    return R_b / continuum


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

    wl = _as_float64(wl_nm)
    R_b = _as_float64(R)

    if wl.ndim != 1:
        raise ValueError("wl_nm must be a one-dimensional wavelength grid")
    if R_b.shape[-1] != wl.size:
        msg = "Last dimension of R must match wavelength grid length"
        raise ValueError(msg)

    spectrum = Spectrum.from_reflectance(WavelengthGrid(wl), R_b)
    flat_vals = spectrum.values.reshape(-1, wl.size)
    depths = np.empty(flat_vals.shape[0], dtype=np.float64)

    for idx, spec_vals in enumerate(flat_vals):
        spec = Spectrum.from_reflectance(WavelengthGrid(wl), spec_vals)
        metrics = compute_band_metrics(
            spec,
            band=BandDefinition(
                lambda_center_nm=float(center_nm),
                lambda_left_nm=float(left_nm),
                lambda_right_nm=float(right_nm),
            ),
            method="anchors",
            anchors=[(float(left_nm), float(right_nm))],
        )
        depths[idx] = metrics.depth

    return depths.reshape(R_b.shape[:-1])
