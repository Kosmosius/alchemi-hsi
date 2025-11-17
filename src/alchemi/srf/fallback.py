"""Gaussian fallback SRF builder and validation utilities."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence

import numpy as np

from alchemi.types import SRFMatrix


def _as_float_array(values: Sequence[float], *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D sequence")
    return arr


def _grid_spacing(wavelengths: np.ndarray) -> float:
    if wavelengths.size < 2:
        raise ValueError("high-resolution wavelength grid must contain at least two samples")
    diffs = np.diff(wavelengths)
    if np.any(diffs <= 0):
        raise ValueError("high-resolution wavelength grid must be strictly increasing")
    return float(np.median(diffs))


def _gaussian_stats(center_nm: float, fwhm_nm: float) -> tuple[float, float]:
    if not np.isfinite(center_nm):
        raise ValueError("Gaussian center must be finite")
    if not np.isfinite(fwhm_nm) or fwhm_nm <= 0:
        raise ValueError("Gaussian FWHM must be positive and finite")
    sigma = float(fwhm_nm) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return float(center_nm), sigma


def gaussian_srf(center_nm: float, fwhm_nm: float, highres_wl_nm: np.ndarray) -> np.ndarray:
    """Generate a normalized Gaussian SRF on ``highres_wl_nm``.

    Parameters
    ----------
    center_nm:
        Center wavelength of the band (nm).
    fwhm_nm:
        Full-width at half-maximum (nm).
    highres_wl_nm:
        High-resolution wavelength grid on which to evaluate the Gaussian (nm).

    Returns
    -------
    np.ndarray
        Normalized response samples aligned with ``highres_wl_nm``.
    """

    wl = _as_float_array(highres_wl_nm, name="highres_wl_nm")
    spacing = _grid_spacing(wl)
    center, sigma = _gaussian_stats(center_nm, fwhm_nm)

    if fwhm_nm < spacing:
        warnings.warn(
            "FWHM is smaller than the wavelength sampling interval; Gaussian may be undersampled.",
            RuntimeWarning,
            stacklevel=2,
        )
    if fwhm_nm > (wl[-1] - wl[0]):
        warnings.warn(
            "FWHM is wider than the wavelength grid span; Gaussian may be poorly supported.",
            RuntimeWarning,
            stacklevel=2,
        )

    exponent = -0.5 * ((wl - center) / sigma) ** 2
    response = np.exp(exponent)

    area = float(np.trapezoid(response, wl))
    if area <= 0:
        raise ValueError("Gaussian SRF integrates to a non-positive area")
    response /= area
    return response


def build_matrix_from_centers(
    centers_nm: Iterable[float],
    fwhm_nm: float | Sequence[float],
    highres_wl_nm: np.ndarray,
    *,
    sensor: str = "gaussian_fallback",
    version: str = "v1",
) -> SRFMatrix:
    """Construct an :class:`~alchemi.types.SRFMatrix` from Gaussian fallbacks."""

    centers = _as_float_array(centers_nm, name="centers_nm")
    wl = _as_float_array(highres_wl_nm, name="highres_wl_nm")
    _ = _grid_spacing(wl)  # validates monotonicity

    if np.isscalar(fwhm_nm):
        fwhms = np.full_like(centers, float(fwhm_nm), dtype=np.float64)
    else:
        fwhms = _as_float_array(fwhm_nm, name="fwhm_nm")
        if fwhms.shape != centers.shape:
            raise ValueError("fwhm_nm array must match centers_nm shape")

    bands_nm = []
    bands_resp = []
    for center, width in zip(centers, fwhms, strict=True):
        resp = gaussian_srf(float(center), float(width), wl)
        bands_nm.append(wl.copy())
        bands_resp.append(resp)

    matrix = SRFMatrix(
        sensor=sensor,
        centers_nm=centers,
        bands_nm=bands_nm,
        bands_resp=bands_resp,
        version=version,
    )
    validate_srf_matrix(matrix)
    return matrix


def validate_srf_matrix(matrix: SRFMatrix) -> None:
    """Validate that SRF rows are normalized and sufficiently supported."""

    if len(matrix.bands_nm) != len(matrix.centers_nm) or len(matrix.bands_resp) != len(
        matrix.centers_nm
    ):
        raise ValueError("SRFMatrix bands must align with centers")

    for idx, (center, wl, resp) in enumerate(
        zip(matrix.centers_nm, matrix.bands_nm, matrix.bands_resp, strict=True)
    ):
        wl_arr = _as_float_array(wl, name=f"bands_nm[{idx}]")
        resp_arr = _as_float_array(resp, name=f"bands_resp[{idx}]")
        if wl_arr.shape != resp_arr.shape:
            raise ValueError("Wavelength and response arrays must share the same shape")

        _ = _grid_spacing(wl_arr)

        area = float(np.trapezoid(resp_arr, wl_arr))
        if not np.isfinite(area) or abs(area - 1.0) > 1e-3:
            raise ValueError(f"SRF row {idx} is not normalized (area={area})")

        mean = float(np.trapezoid(wl_arr * resp_arr, wl_arr))
        var = float(np.trapezoid(((wl_arr - mean) ** 2) * resp_arr, wl_arr))
        sigma = float(np.sqrt(max(var, 0.0)))

        if not np.isfinite(mean) or not np.isfinite(sigma) or sigma <= 0:
            raise ValueError(f"SRF row {idx} has invalid statistics")

        if abs(mean - center) > 0.5:
            raise ValueError(
                f"SRF row {idx} mean {mean:.3f} nm deviates from center {float(center):.3f} nm"
            )

        support_min = wl_arr[0]
        support_max = wl_arr[-1]
        required_min = float(center) - 3.0 * sigma
        required_max = float(center) + 3.0 * sigma
        tol = 1e-6
        if support_min > required_min + tol or support_max < required_max - tol:
            raise ValueError(
                "SRF row {idx} support [{support_min:.3f}, {support_max:.3f}] nm does not cover "
                "3-sigma interval"
            )

    return None
