"""Spectral resampling utilities with SRF-aware and SRF-free fallbacks."""

from __future__ import annotations

from typing import Literal

import numpy as np

from alchemi.utils.integrate import np_integrate as _np_integrate

from alchemi.types import SRFMatrix

__all__ = [
    "boxcar_resample",
    "convolve_to_bands",
    "gaussian_resample",
    "project_to_sensor",
]


def _validate_wavelengths(wl_nm: np.ndarray) -> np.ndarray:
    wl = np.asarray(wl_nm, dtype=np.float64)
    if wl.ndim != 1:
        raise ValueError("Wavelength grid must be 1-D")
    if wl.size < 2 or np.any(np.diff(wl) <= 0):
        raise ValueError("Wavelength grid must be strictly increasing")
    return wl


def _as_2d(values: np.ndarray, n_wavelengths: int) -> tuple[np.ndarray, bool]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape[0] != n_wavelengths:
            raise ValueError("Spectrum length must match wavelength grid")
        return arr[None, :], True
    if arr.ndim == 2:
        if arr.shape[1] != n_wavelengths:
            raise ValueError("Spectrum length must match wavelength grid")
        return arr, False
    raise ValueError("Spectral values must be a 1-D or 2-D array")


def convolve_to_bands(
    highres_wl_nm: np.ndarray,
    highres_vals: np.ndarray,
    srf_matrix: SRFMatrix,
) -> np.ndarray:
    """Convolve a high-resolution spectrum with tabulated SRFs."""

    wl = _validate_wavelengths(np.asarray(highres_wl_nm, dtype=np.float64))
    spectra, squeezed = _as_2d(np.asarray(highres_vals, dtype=np.float64), wl.shape[0])

    n_bands = len(srf_matrix.centers_nm)
    out = np.empty((spectra.shape[0], n_bands), dtype=np.float64)

    for idx, (band_wl, band_resp) in enumerate(
        zip(srf_matrix.bands_nm, srf_matrix.bands_resp, strict=True)
    ):
        band_wl = np.asarray(band_wl, dtype=np.float64)
        band_resp = np.asarray(band_resp, dtype=np.float64)
        if band_wl.ndim != 1 or band_resp.ndim != 1:
            raise ValueError("SRF band definitions must be 1-D arrays")
        if band_wl.shape[0] != band_resp.shape[0]:
            raise ValueError("SRF band wavelength/response length mismatch")

        interp_vals = np.vstack([np.interp(band_wl, wl, row) for row in spectra])
        weighted = interp_vals * band_resp[None, :]
        band_area = float(_np_integrate(band_resp, band_wl))
        if not np.isfinite(band_area) or band_area <= 0.0:
            raise ValueError("SRF band must integrate to a positive finite area")
        out[:, idx] = _np_integrate(weighted, band_wl, axis=1) / band_area

    return out[0] if squeezed else out


def boxcar_resample(
    wl_nm: np.ndarray,
    vals: np.ndarray,
    target_centers_nm: np.ndarray,
    width_nm: np.ndarray | float,
) -> np.ndarray:
    """Resample spectra using normalized boxcar kernels centered at target wavelengths."""

    wl = _validate_wavelengths(wl_nm)
    spectra, squeezed = _as_2d(vals, wl.shape[0])

    centers = np.asarray(target_centers_nm, dtype=np.float64)
    if centers.ndim != 1:
        raise ValueError("Target band centers must be 1-D")

    widths = np.broadcast_to(np.asarray(width_nm, dtype=np.float64), centers.shape)
    if np.any(widths <= 0):
        raise ValueError("Boxcar widths must be positive")

    out = np.empty((spectra.shape[0], centers.shape[0]), dtype=np.float64)

    for idx, (center, width) in enumerate(zip(centers, widths, strict=True)):
        half_width = 0.5 * width
        lower = center - half_width
        upper = center + half_width

        interior = wl[(wl > lower) & (wl < upper)]
        window_wl = np.concatenate(([lower], interior, [upper]))
        interp_vals = np.vstack([np.interp(window_wl, wl, row) for row in spectra])
        out[:, idx] = _np_integrate(interp_vals, window_wl, axis=1) / width

    return out[0] if squeezed else out


def gaussian_resample(
    wl_nm: np.ndarray,
    vals: np.ndarray,
    target_centers_nm: np.ndarray,
    fwhm_nm: np.ndarray | float,
) -> np.ndarray:
    """Resample spectra using normalized Gaussian kernels centered at target wavelengths."""

    wl = _validate_wavelengths(wl_nm)
    spectra, squeezed = _as_2d(vals, wl.shape[0])

    centers = np.asarray(target_centers_nm, dtype=np.float64)
    if centers.ndim != 1:
        raise ValueError("Target band centers must be 1-D")

    fwhm = np.broadcast_to(np.asarray(fwhm_nm, dtype=np.float64), centers.shape)
    if np.any(fwhm <= 0):
        raise ValueError("Gaussian FWHM values must be positive")

    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    out = np.empty((spectra.shape[0], centers.shape[0]), dtype=np.float64)

    for idx, (center, sig) in enumerate(zip(centers, sigma, strict=True)):
        weights = np.exp(-0.5 * ((wl - center) / sig) ** 2)
        denom = float(_np_integrate(weights, wl))
        if not np.isfinite(denom) or denom <= 0.0:
            raise ValueError("Gaussian kernel must integrate to a positive finite area")
        weighted = spectra * weights[None, :]
        out[:, idx] = _np_integrate(weighted, wl, axis=1) / denom

    return out[0] if squeezed else out


def project_to_sensor(
    wl_nm: np.ndarray,
    vals: np.ndarray,
    target_centers_nm: np.ndarray,
    *,
    srf: SRFMatrix | None = None,
    fwhm_nm: np.ndarray | float | None = None,
    width_nm: np.ndarray | float | None = None,
    fallback: Literal["gaussian", "box"] = "gaussian",
) -> np.ndarray:
    """Project spectra onto sensor bands using SRFs or analytic fallbacks."""

    centers = np.asarray(target_centers_nm, dtype=np.float64)
    if centers.ndim != 1:
        raise ValueError("Target band centers must be 1-D")

    if srf is not None:
        if srf.centers_nm.shape[0] != centers.shape[0]:
            raise ValueError("SRF band count does not match target centers")
        if not np.allclose(srf.centers_nm, centers):
            raise ValueError("SRF centers do not align with target centers")
        return convolve_to_bands(wl_nm, vals, srf)

    if fallback == "gaussian":
        if fwhm_nm is None:
            if width_nm is not None:
                fwhm_nm = width_nm
            else:
                raise ValueError("Gaussian fallback requires FWHM information")
        return gaussian_resample(wl_nm, vals, centers, fwhm_nm)

    if fallback == "box":
        if width_nm is None:
            if fwhm_nm is not None:
                width_nm = fwhm_nm
            else:
                raise ValueError("Boxcar fallback requires width information")
        return boxcar_resample(wl_nm, vals, centers, width_nm)

    raise ValueError(f"Unsupported fallback method '{fallback}'")
