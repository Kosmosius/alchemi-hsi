"""Spectral resampling utilities (convolution, interpolation, virtual sensors)."""

from __future__ import annotations

from typing import Literal

import numpy as np

from alchemi.spectral.srf import SRFMatrix as DenseSRFMatrix
from alchemi.types import Spectrum, SRFMatrix, WavelengthGrid
from alchemi.utils.integrate import np_integrate

__all__ = [
    "convolve_to_bands",
    "convolve_to_bands_batched",
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
    """Convolve a high-resolution spectrum to sensor bands using SRF rows.

    For batched or dense SRF workflows, prefer
    :func:`convolve_to_bands_batched`.
    """

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
        wavelengths=WavelengthGrid(centers),
        values=band_array,
        kind=high_res.kind,
        units=high_res.units,
        mask=srf.bad_band_mask,
        meta=high_res.meta.copy(),
    )


def _as_dense_srf_matrix(
    srf_matrix: np.ndarray | DenseSRFMatrix,
    srf_wavelength_nm: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(srf_matrix, DenseSRFMatrix):
        wavelength = np.asarray(srf_matrix.wavelength_nm, dtype=np.float64)
        matrix = np.asarray(srf_matrix.matrix, dtype=np.float64)
    else:
        matrix = np.asarray(srf_matrix, dtype=np.float64)
        if matrix.ndim != 2:
            msg = "srf_matrix must be 2-D (bands x wavelengths)"
            raise ValueError(msg)
        if srf_wavelength_nm is None:
            msg = "srf_wavelength_nm must be provided when srf_matrix is an array"
            raise ValueError(msg)
        wavelength = np.asarray(srf_wavelength_nm, dtype=np.float64)

    if wavelength.ndim != 1:
        msg = "srf_wavelength_nm must be 1-D"
        raise ValueError(msg)
    if matrix.shape[1] != wavelength.shape[0]:
        msg = "srf_matrix column count must match srf_wavelength_nm length"
        raise ValueError(msg)
    return matrix, wavelength


def convolve_to_bands_batched(
    spectra: np.ndarray,
    wavelengths_nm: np.ndarray,
    srf_matrix: np.ndarray | DenseSRFMatrix,
    srf_wavelength_nm: np.ndarray | None = None,
) -> np.ndarray:
    """Vectorized SRF convolution for batches or cubes of high-res spectra."""

    spectra_arr = np.asarray(spectra, dtype=np.float64)
    wavelengths = np.asarray(wavelengths_nm, dtype=np.float64)
    if spectra_arr.ndim < 1:
        raise ValueError("spectra must have at least one dimension")
    if wavelengths.ndim != 1:
        raise ValueError("wavelengths_nm must be 1-D")
    if spectra_arr.shape[-1] != wavelengths.shape[0]:
        msg = "spectra last dimension must match wavelengths_nm length"
        raise ValueError(msg)

    dense_srf, srf_wavelength = _as_dense_srf_matrix(srf_matrix, srf_wavelength_nm)
    if dense_srf.ndim != 2:
        raise ValueError("srf_matrix must be 2-D (bands x wavelengths)")

    orig_shape = spectra_arr.shape[:-1]
    flat_spectra = spectra_arr.reshape(-1, spectra_arr.shape[-1])

    if not np.array_equal(wavelengths, srf_wavelength):
        interpolated = np.empty((flat_spectra.shape[0], srf_wavelength.shape[0]))
        for idx, row in enumerate(flat_spectra):
            interpolated[idx] = np.interp(srf_wavelength, wavelengths, row)
        flat_spectra = interpolated
    else:
        flat_spectra = flat_spectra.copy()

    integrals = np.trapz(dense_srf, x=srf_wavelength, axis=1)
    if np.any(~np.isfinite(integrals)) or np.any(integrals <= 0.0):
        msg = "SRF rows must integrate to a positive finite area"
        raise ValueError(msg)
    resp_norm = dense_srf / integrals[:, np.newaxis]

    product = resp_norm[np.newaxis, :, :] * flat_spectra[:, np.newaxis, :]
    band_values = np.trapz(product, x=srf_wavelength, axis=2)

    return band_values.reshape(*orig_shape, dense_srf.shape[0])


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
        wavelengths=WavelengthGrid(centers),
        values=np.asarray(interp_vals, dtype=np.float64),
        kind=high_res.kind,
        units=high_res.units,
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
