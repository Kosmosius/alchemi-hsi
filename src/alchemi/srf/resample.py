"""Spectral resampling utilities with SRF-aware and SRF-free fallbacks."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from alchemi.physics import resampling as physics_resampling
from alchemi.types import QuantityKind, SRFMatrix, Spectrum, ValueUnits, WavelengthGrid
from alchemi.utils.integrate import np_integrate as _np_integrate

__all__ = [
    "boxcar_resample",
    "convolve_to_bands",
    "gaussian_resample",
    "interpolate_values",
    "resample_by_interpolation",
    "resample_with_srf",
    "project_to_sensor",
    "resample_values_with_srf",
    "resample_to_sensor",
]


def _validate_wavelengths(wl_nm: np.ndarray) -> NDArray[np.float64]:
    wl = np.asarray(wl_nm, dtype=np.float64)
    if wl.ndim != 1:
        raise ValueError("Wavelength grid must be 1-D")
    if wl.size < 2 or np.any(np.diff(wl) <= 0):
        raise ValueError("Wavelength grid must be strictly increasing")
    return wl


def _grid_deltas(wl_nm: NDArray[np.float64]) -> NDArray[np.float64]:
    diffs = np.diff(wl_nm)
    deltas = np.empty_like(wl_nm)
    deltas[0] = diffs[0]
    deltas[-1] = diffs[-1]
    if wl_nm.size > 2:
        deltas[1:-1] = 0.5 * (diffs[:-1] + diffs[1:])
    return deltas


def _as_2d(values: np.ndarray, n_wavelengths: int) -> tuple[NDArray[np.float64], bool]:
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


def _to_srf_matrix(srf: SRFMatrix | object) -> SRFMatrix:
    if isinstance(srf, SRFMatrix):
        return srf

    wl = getattr(srf, "wavelength_grid_nm", None)
    rows = getattr(srf, "srfs", None)
    if wl is None or rows is None:
        msg = "srf_matrix must be an SRFMatrix or SensorSRF-like object"
        raise TypeError(msg)

    wl_arr = _validate_wavelengths(np.asarray(wl, dtype=np.float64))
    row_arr = np.asarray(rows, dtype=np.float64)
    if row_arr.ndim != 2 or row_arr.shape[1] != wl_arr.shape[0]:
        msg = "srfs must be shaped [bands, wavelengths]"
        raise ValueError(msg)

    centers = getattr(srf, "band_centers_nm", None)

    band_centers: list[np.ndarray] = []
    bands_resp: list[np.ndarray] = []
    bands_nm: list[np.ndarray] = []
    for row in row_arr:
        positive = np.nonzero(row > 0.0)[0]
        if positive.size == 0:
            msg = "SRF rows must contain positive support"
            raise ValueError(msg)
        start, end = positive[0], positive[-1]

        nm_band = wl_arr[start : end + 1]
        resp_band = row[start : end + 1]

        deltas_band = _grid_deltas(nm_band)
        row_scale = float(resp_band @ deltas_band)
        if not np.isfinite(row_scale) or row_scale <= 0.0:
            msg = "SRF rows must integrate to a positive finite area"
            raise ValueError(msg)

        trapz_area = float(np.trapezoid(resp_band, x=nm_band))
        if not np.isfinite(trapz_area) or trapz_area <= 0.0:
            msg = "SRF rows must have positive area"
            raise ValueError(msg)

        scaled_row = np.asarray(resp_band * (row_scale / trapz_area), dtype=np.float64)
        bands_resp.append(scaled_row)
        band_centers.append((scaled_row * deltas_band) @ nm_band / row_scale)
        bands_nm.append(nm_band)

    centers_arr = np.asarray(centers if centers is not None else band_centers, dtype=np.float64)
    bands_nm = [np.asarray(nm_band, dtype=np.float64) for nm_band in bands_nm]
    meta = getattr(srf, "meta", {}) or {}

    return SRFMatrix(
        sensor=str(meta.get("sensor", getattr(srf, "sensor_id", "sensor"))),
        centers_nm=centers_arr,
        bands_nm=bands_nm,
        bands_resp=bands_resp,
        version=str(meta.get("srf_version", meta.get("version", "v1"))),
        bad_band_mask=getattr(srf, "valid_mask", None) or meta.get("bad_band_mask"),
        bad_band_windows_nm=meta.get("bad_band_windows_nm"),
    )


def resample_values_with_srf(
    values: np.ndarray,
    wavelengths_hr_nm: np.ndarray,
    srf_matrix: SRFMatrix,
    *,
    allow_mismatch_tol_nm: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample raw arrays onto sensor bands using SRF weights."""

    srf_matrix = _to_srf_matrix(srf_matrix)
    _ = allow_mismatch_tol_nm  # retained for API compatibility

    wl_hr = _validate_wavelengths(wavelengths_hr_nm)
    spectra, squeezed = _as_2d(values, wl_hr.shape[0])
    band_values: list[np.ndarray] = []

    for row in spectra:
        spec = Spectrum(
            wavelengths=WavelengthGrid(wl_hr),
            values=np.asarray(row, dtype=np.float64),
            kind=QuantityKind.RADIANCE,
            units=ValueUnits.RADIANCE_W_M2_SR_NM,
        )
        convolved = physics_resampling.convolve_to_bands(spec, srf_matrix)
        band_values.append(np.asarray(convolved.values, dtype=np.float64))

    band_array = np.vstack(band_values)
    centers = np.asarray(srf_matrix.centers_nm, dtype=np.float64)
    result = band_array[0] if squeezed else band_array
    return result, centers


def resample_with_srf(
    spectrum: Spectrum,
    srf_matrix: SRFMatrix,
    *,
    allow_mismatch_tol_nm: float = 1e-3,
) -> Spectrum:
    """Project a high-resolution :class:`Spectrum` onto sensor bandspace."""

    band_values, centers = resample_values_with_srf(
        spectrum.values,
        spectrum.wavelengths.nm,
        srf_matrix,
        allow_mismatch_tol_nm=allow_mismatch_tol_nm,
    )

    meta = dict(spectrum.meta)
    if "band_centers_nm" not in meta:
        meta["band_centers_nm"] = centers

    return Spectrum(
        wavelengths=WavelengthGrid(centers),
        values=np.asarray(band_values, dtype=np.float64),
        kind=spectrum.kind,
        units=spectrum.units,
        meta=meta,
    )


def resample_to_sensor(
    spectrum: Spectrum, sensor_id: str, **kwargs: float
) -> Spectrum:
    from .utils import load_sensor_srf

    srf_matrix = load_sensor_srf(sensor_id)
    if srf_matrix is None:
        msg = f"No SRF matrix available for sensor_id={sensor_id!r}"
        raise KeyError(msg)
    return resample_with_srf(spectrum, srf_matrix, **kwargs)


def convolve_to_bands(
    highres_wl_nm: np.ndarray,
    highres_vals: np.ndarray,
    srf_matrix: SRFMatrix,
) -> NDArray[np.float64]:
    """Convolve a high-resolution spectrum with tabulated SRFs."""

    srf_matrix = _to_srf_matrix(srf_matrix)
    wl = _validate_wavelengths(np.asarray(highres_wl_nm, dtype=np.float64))
    spectrum = Spectrum(
        wavelengths=WavelengthGrid(wl),
        values=np.asarray(highres_vals, dtype=np.float64),
        kind=QuantityKind.UNKNOWN,
        units=None,
    )
    convolved = physics_resampling.convolve_to_bands(spectrum, srf_matrix)
    return np.asarray(convolved.values, dtype=np.float64)


def boxcar_resample(
    wl_nm: np.ndarray,
    vals: np.ndarray,
    target_centers_nm: np.ndarray,
    width_nm: np.ndarray | float,
) -> NDArray[np.float64]:
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

    result = out[0] if squeezed else out
    return np.asarray(result, dtype=np.float64)


def gaussian_resample(
    wl_nm: np.ndarray,
    vals: np.ndarray,
    target_centers_nm: np.ndarray,
    fwhm_nm: np.ndarray | float,
) -> NDArray[np.float64]:
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

    result = out[0] if squeezed else out
    return np.asarray(result, dtype=np.float64)


def interpolate_values(
    values: np.ndarray,
    wavelengths_nm: np.ndarray,
    target_centers_nm: np.ndarray,
    *,
    mode: Literal["nearest", "linear", "spline"] = "linear",
) -> NDArray[np.float64]:
    """Interpolate spectral values onto target centers.

    Parameters
    ----------
    values:
        Spectral values shaped ``[L]`` or ``[N, L]`` where ``L`` is the number
        of wavelengths.
    wavelengths_nm:
        Source wavelength grid in **nanometres**, strictly increasing and
        matching the last dimension of ``values``.
    target_centers_nm:
        Target wavelength coordinates (nm) to sample at. This function performs
        *center-based interpolation* only; no bandpass integration is applied.
    mode:
        Interpolation strategy: ``"nearest"``, ``"linear"``, or ``"spline"``.
        ``"spline"`` uses SciPy's :class:`~scipy.interpolate.CubicSpline` when
        available and raises ``RuntimeError`` otherwise.
    """

    wl = _validate_wavelengths(np.asarray(wavelengths_nm, dtype=np.float64))
    spectra, squeezed = _as_2d(np.asarray(values, dtype=np.float64), wl.shape[0])

    centers = np.asarray(target_centers_nm, dtype=np.float64)
    if centers.ndim != 1:
        raise ValueError("Target band centers must be 1-D")

    mode = str(mode).lower()
    if mode not in {"nearest", "linear", "spline"}:
        raise ValueError("mode must be one of 'nearest', 'linear', or 'spline'")

    if mode == "nearest":
        right_idx = np.searchsorted(wl, centers, side="left")
        left_idx = np.clip(right_idx - 1, 0, wl.shape[0] - 1)
        right_idx = np.clip(right_idx, 0, wl.shape[0] - 1)
        choose_right = np.abs(wl[right_idx] - centers) <= np.abs(wl[left_idx] - centers)
        indices = np.where(choose_right, right_idx, left_idx)
        out = spectra[:, indices]
    elif mode == "linear":
        out = np.vstack([np.interp(centers, wl, row) for row in spectra])
    else:
        try:
            from scipy.interpolate import CubicSpline
        except Exception as exc:  # pragma: no cover - exercised via tests
            msg = (
                "Spline interpolation requires SciPy; install scipy or choose "
                "mode='linear'"
            )
            raise RuntimeError(msg) from exc

        out = np.vstack([CubicSpline(wl, row, extrapolate=True)(centers) for row in spectra])

    result = out[0] if squeezed else out
    return np.asarray(result, dtype=np.float64)


def project_to_sensor(
    wl_nm: np.ndarray,
    vals: np.ndarray,
    target_centers_nm: np.ndarray,
    *,
    srf: SRFMatrix | None = None,
    fwhm_nm: np.ndarray | float | None = None,
    width_nm: np.ndarray | float | None = None,
    fallback: Literal["gaussian", "box"] = "gaussian",
) -> NDArray[np.float64]:
    """Project spectra onto sensor bands using SRFs or analytic fallbacks."""

    centers = np.asarray(target_centers_nm, dtype=np.float64)
    if centers.ndim != 1:
        raise ValueError("Target band centers must be 1-D")

    if srf is not None:
        if srf.centers_nm.shape[0] != centers.shape[0]:
            raise ValueError("SRF band count does not match target centers")
        if not np.allclose(srf.centers_nm, centers):
            raise ValueError("SRF centers do not align with target centers")
        band_values, _ = resample_values_with_srf(vals, wl_nm, srf)
        return band_values

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


def resample_by_interpolation(
    spectrum: Spectrum,
    target_centers_nm: np.ndarray,
    *,
    mode: Literal["nearest", "linear", "spline"] = "linear",
) -> Spectrum:
    """Resample a :class:`~alchemi.spectral.Spectrum` by band-center interpolation.

    This performs *center sampling* only; no SRF/bandpass convolution is
    applied. Callers must choose this explicitly when SRFs are unavailable or
    intentionally ignored.
    """

    new_values = interpolate_values(
        spectrum.values,
        spectrum.wavelengths.nm,
        target_centers_nm,
        mode=mode,
    )

    target_grid = WavelengthGrid.from_any(target_centers_nm, units="nm")
    meta = dict(spectrum.meta)
    meta["resample_mode"] = "center_interp"
    meta["interp_mode"] = mode

    return Spectrum(
        wavelengths=target_grid,
        values=new_values,
        kind=spectrum.kind,
        units=spectrum.units,
        mask=None,
        meta=meta,
    )
