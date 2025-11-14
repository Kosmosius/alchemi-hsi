"""Spectral response utilities for the Mako LWIR sensor."""

from __future__ import annotations

import hashlib

import numpy as np

from alchemi.types import SRFMatrix

try:
    from numpy import trapezoid as _integrate
except ImportError:  # pragma: no cover - NumPy < 2.0 fallback
    from numpy import trapz as _integrate  # type: ignore[attr-defined]

__all__ = [
    "MAKO_BAND_COUNT",
    "build_mako_srf_from_header",
    "mako_lwir_grid_nm",
]


MAKO_BAND_COUNT = 128
_SENSOR_ID = "mako"
_DEFAULT_VERSION = "comex-l2s-gaussian-v1"
_DEFAULT_FWHM_NM = 44.0
_GRID_START_NM = 7400.0
_GRID_STOP_NM = 13600.0
_GRID_STEP_NM = 10.0


def mako_lwir_grid_nm() -> np.ndarray:
    """Return the canonical LWIR wavelength grid used for Mako SRFs (nm)."""

    span = _GRID_STOP_NM - _GRID_START_NM
    count = int(round(span / _GRID_STEP_NM)) + 1
    grid = np.linspace(_GRID_START_NM, _GRID_STOP_NM, count, dtype=np.float64)
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError("LWIR grid must be a strictly increasing 1-D array")
    return grid


def _validate_header_wavelengths(wavelengths_nm: np.ndarray) -> np.ndarray:
    centers = np.asarray(wavelengths_nm, dtype=np.float64)
    if centers.ndim != 1:
        raise ValueError("Header wavelengths must be a 1-D array")
    if centers.size == 0:
        raise ValueError("Header wavelengths must contain at least one value")
    if np.any(np.diff(centers) <= 0.0):
        raise ValueError("Header wavelengths must be strictly increasing")
    return centers


def _compute_sigma(fwhm_nm: float) -> float:
    if not np.isfinite(fwhm_nm) or fwhm_nm <= 0.0:
        raise ValueError("FWHM must be a positive finite value")
    return fwhm_nm / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def _compute_cache_key(
    centers_nm: np.ndarray, grid_nm: np.ndarray, fwhm_nm: float, version: str
) -> str:
    hasher = hashlib.sha1()
    hasher.update(_SENSOR_ID.encode("utf-8"))
    hasher.update(version.encode("utf-8"))
    hasher.update(np.asarray(centers_nm, dtype=np.float64).tobytes())
    hasher.update(np.asarray(grid_nm, dtype=np.float64).tobytes())
    hasher.update(np.float64(fwhm_nm).tobytes())
    return f"{_SENSOR_ID}:{version}:{hasher.hexdigest()[:12]}"


def build_mako_srf_from_header(
    wavelengths_nm: np.ndarray, fwhm_nm: float = _DEFAULT_FWHM_NM
) -> SRFMatrix:
    """Construct a trapz-normalised SRF matrix from COMEX header wavelengths."""

    centers_nm = _validate_header_wavelengths(wavelengths_nm)
    sigma = _compute_sigma(float(fwhm_nm))

    grid_nm = mako_lwir_grid_nm()
    bands_nm: list[np.ndarray] = []
    bands_resp: list[np.ndarray] = []

    for center in centers_nm:
        response = np.exp(-0.5 * ((grid_nm - center) / sigma) ** 2)
        area = float(_integrate(response, grid_nm))
        if not np.isfinite(area) or area <= 0.0:
            raise ValueError("SRF band must integrate to a positive finite area")
        bands_nm.append(grid_nm.copy())
        bands_resp.append(response / area)

    srf = SRFMatrix(_SENSOR_ID, centers_nm, bands_nm, bands_resp, version=_DEFAULT_VERSION)
    srf = srf.normalize_trapz()
    srf.cache_key = _compute_cache_key(centers_nm, grid_nm, float(fwhm_nm), srf.version)
    return srf

