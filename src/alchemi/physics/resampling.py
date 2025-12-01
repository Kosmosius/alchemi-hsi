"""Spectral resampling utilities (convolution, interpolation, virtual sensors)."""

from __future__ import annotations

from typing import Literal

import numpy as np

from alchemi.types import Spectrum, SRFMatrix, WavelengthGrid
from alchemi.utils.integrate import np_integrate

__all__ = [
    "convolve_to_bands",
    "interpolate_to_centers",
    "generate_gaussian_srf",
    "simulate_virtual_sensor",
]


def _normalize_response(resp: np.ndarray, nm: np.ndarray) -> np.ndarray:
    area = float(np_integrate(resp, nm))
    if not np.isfinite(area) or area <= 0.0:
        raise ValueError("SRF responses must integrate to a positive value")
    return resp / area


def convolve_to_bands(high_res: Spectrum, srf: SRFMatrix) -> Spectrum:
    """Convolve a high-resolution spectrum to sensor bands using SRF rows."""

    nm_hi = np.asarray(high_res.wavelengths.nm, dtype=np.float64)
    values_hi = np.asarray(high_res.values, dtype=np.float64)
    band_values: list[float] = []

    for nm_band, resp in zip(srf.bands_nm, srf.bands_resp, strict=True):
        nm_arr = np.asarray(nm_band, dtype=np.float64)
        resp_arr = np.asarray(resp, dtype=np.float64)
        resp_norm = _normalize_response(resp_arr, nm_arr)
        interp_vals = np.interp(nm_arr, nm_hi, values_hi)
        band_value = float(np_integrate(interp_vals * resp_norm, nm_arr))
        band_values.append(band_value)

    band_array = np.asarray(band_values, dtype=np.float64)
    centers = np.asarray(srf.centers_nm, dtype=np.float64)

    return Spectrum(
        WavelengthGrid(centers),
        band_array,
        high_res.kind,
        high_res.units,
        mask=srf.bad_band_mask,
        meta=high_res.meta.copy(),
    )


def interpolate_to_centers(
    high_res: Spectrum,
    centers_nm: np.ndarray,
    mode: Literal["nearest", "linear", "spline"] = "linear",
) -> Spectrum:
    """Interpolate a spectrum to specified band centres."""

    centers = np.asarray(centers_nm, dtype=np.float64)
    nm_hi = np.asarray(high_res.wavelengths.nm, dtype=np.float64)
    values_hi = np.asarray(high_res.values, dtype=np.float64)

    if mode not in {"nearest", "linear", "spline"}:
        raise ValueError("mode must be 'nearest', 'linear', or 'spline'")

    if mode == "nearest":
        idx = np.searchsorted(nm_hi, centers, side="left")
        idx = np.clip(idx, 0, nm_hi.size - 1)
        interp_vals = values_hi[idx]
    elif mode == "linear":
        interp_vals = np.interp(centers, nm_hi, values_hi)
    else:  # spline
        try:
            from scipy.interpolate import CubicSpline
        except Exception:  # pragma: no cover - optional dependency
            interp_vals = np.interp(centers, nm_hi, values_hi)
        else:
            spline = CubicSpline(nm_hi, values_hi, extrapolate=True)
            interp_vals = spline(centers)

    return Spectrum(
        WavelengthGrid(centers),
        np.asarray(interp_vals, dtype=np.float64),
        high_res.kind,
        high_res.units,
        mask=None,
        meta=high_res.meta.copy(),
    )


def generate_gaussian_srf(
    sensor_name: str,
    range_nm: tuple[float, float],
    num_bands: int,
    fwhm_nm: float | None = None,
) -> SRFMatrix:
    """Generate a Gaussian SRF grid across a spectral range."""

    start, end = range_nm
    centers = np.linspace(start, end, num_bands)
    if fwhm_nm is None:
        fwhm_nm = (end - start) / max(num_bands, 1)
    sigma = fwhm_nm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    bands_nm: list[np.ndarray] = []
    bands_resp: list[np.ndarray] = []
    for center in centers:
        nm = np.linspace(center - 3 * sigma, center + 3 * sigma, 61)
        resp = np.exp(-0.5 * ((nm - center) / sigma) ** 2)
        bands_nm.append(nm)
        bands_resp.append(_normalize_response(resp, nm))

    return SRFMatrix(sensor_name, centers, bands_nm, bands_resp)


def simulate_virtual_sensor(
    high_res: Spectrum,
    range_nm: tuple[float, float],
    num_bands: int,
    fwhm_nm: float | None = None,
    sensor_name: str = "virtual_sensor",
) -> Spectrum:
    """Generate a virtual sensor SRF and convolve the provided spectrum."""

    srf = generate_gaussian_srf(sensor_name, range_nm, num_bands, fwhm_nm)
    return convolve_to_bands(high_res, srf)
