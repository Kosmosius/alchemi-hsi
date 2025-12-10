"""Spectral resampling utilities (convolution, interpolation, virtual sensors)."""

from __future__ import annotations

from typing import Literal

import numpy as np

from alchemi.spectral.srf import SRFMatrix as DenseSRFMatrix
from alchemi.types import Spectrum, SRFMatrix, WavelengthGrid
from alchemi.utils.integrate import np_integrate
from alchemi.wavelengths import check_monotonic

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


def _compute_delta_wavelength(
    wavelengths_nm: np.ndarray, delta_lambda_nm: np.ndarray | float | None
) -> np.ndarray:
    """Return per-sample wavelength intervals for integration weights."""

    wl = np.asarray(wavelengths_nm, dtype=np.float64)
    check_monotonic(wl, strict=True)

    if delta_lambda_nm is None:
        if wl.size == 1:
            return np.ones_like(wl)

        deltas = np.empty_like(wl)
        step = np.diff(wl)
        deltas[0] = step[0]
        deltas[-1] = step[-1]
        if wl.size > 2:
            deltas[1:-1] = 0.5 * (step[1:] + step[:-1])
    else:
        deltas = np.asarray(delta_lambda_nm, dtype=np.float64)
        deltas = np.broadcast_to(deltas, wl.shape)

    if deltas.shape != wl.shape:
        msg = "delta_lambda_nm must broadcast to wavelength grid shape"
        raise ValueError(msg)
    if np.any(~np.isfinite(deltas)):
        raise ValueError("delta_lambda_nm must be finite")
    if np.any(deltas <= 0.0):
        raise ValueError("delta_lambda_nm values must be positive")

    return deltas


def convolve_to_bands(
    high_res: Spectrum, srf: SRFMatrix, *, delta_lambda_nm: np.ndarray | float | None = None
) -> Spectrum:
    """Convolve a high-resolution spectrum to sensor bands using SRF rows.

    For batched or dense SRF workflows, prefer
    :func:`convolve_to_bands_batched`.
    """

    nm_hi = np.asarray(high_res.wavelengths.nm, dtype=np.float64)
    values_hi = np.asarray(high_res.values, dtype=np.float64)

    dense_rows: list[np.ndarray] = []
    for nm_band, resp in zip(srf.bands_nm, srf.bands_resp, strict=True):
        nm_arr = np.asarray(nm_band, dtype=np.float64)
        resp_arr = np.asarray(resp, dtype=np.float64)
        if resp_arr.ndim == 0:
            resp_arr = resp_arr[np.newaxis]
        interp_resp = np.interp(nm_hi, nm_arr, resp_arr, left=0.0, right=0.0)
        dense_rows.append(_normalize_response(interp_resp, nm_hi))

    dense_matrix = DenseSRFMatrix(wavelength_nm=nm_hi, matrix=np.vstack(dense_rows))
    band_array = convolve_to_bands_batched(
        values_hi[np.newaxis, :], nm_hi, dense_matrix, delta_lambda_nm=delta_lambda_nm
    )[0]

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
    *,
    delta_lambda_nm: np.ndarray | float | None = None,
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

    check_monotonic(wavelengths, strict=True)
    check_monotonic(srf_wavelength, strict=True)

    orig_shape = spectra_arr.shape[:-1]
    flat_spectra = spectra_arr.reshape(-1, spectra_arr.shape[-1])

    if not np.array_equal(wavelengths, srf_wavelength):
        regridded = np.empty((dense_srf.shape[0], wavelengths.shape[0]))
        for idx, row in enumerate(dense_srf):
            regridded[idx] = np.interp(wavelengths, srf_wavelength, row, left=0.0, right=0.0)
        dense_srf = regridded

    delta_lambda = _compute_delta_wavelength(wavelengths, delta_lambda_nm)

    weighted_resp = dense_srf * delta_lambda[np.newaxis, :]
    row_sums = np.sum(weighted_resp, axis=1)
    if np.any(~np.isfinite(row_sums)):
        raise ValueError("SRF weights must sum to finite values")
    if np.any(row_sums <= 0.0):
        msg = "SRF rows must integrate to a positive area when weighted"
        raise ValueError(msg)

    weights = weighted_resp / row_sums[:, np.newaxis]

    band_values = flat_spectra @ weights.T

    return band_values.reshape(*orig_shape, dense_srf.shape[0])


def interpolate_to_centers(
    source_wavelengths_nm: Spectrum | np.ndarray,
    source_values: np.ndarray | None = None,
    target_centers_nm: np.ndarray | None = None,
    mode: Literal["nearest", "linear", "spline"] = "linear",
    *,
    fill_value: float | np.ndarray = np.nan,
    allow_extrapolation: bool = False,
    spline_kwargs: dict | None = None,
) -> Spectrum | np.ndarray:
    """Interpolate spectra to target centres using simple diagnostics-focused modes.

    This helper is intended for exploratory or diagnostic work when sensor
    response functions (SRFs) are unavailable or bandpass effects are
    secondary. For production pipelines with available SRFs, prefer the
    convolution-based routines (:func:`convolve_to_bands` and
    :func:`convolve_to_bands_batched`) to capture bandpass effects.

    Parameters
    ----------
    source_wavelengths_nm:
        Either a high-resolution :class:`~alchemi.types.Spectrum` or a
        one-dimensional wavelength grid in nanometres.
    source_values:
        Spectral values aligned to ``source_wavelengths_nm``. When a
        :class:`~alchemi.types.Spectrum` is provided as the first argument,
        this parameter is treated as ``target_centers_nm`` for backwards
        compatibility and may be ``None`` otherwise.
    target_centers_nm:
        Wavelength centres to which the spectrum should be interpolated. Must
        be provided when using the array-based calling convention.
    mode:
        Interpolation kernel: ``"nearest"``, ``"linear"``, or ``"spline"``.
    fill_value:
        Value used outside the source wavelength range when
        ``allow_extrapolation`` is ``False``. Defaults to ``np.nan`` for clear
        signalling.
    allow_extrapolation:
        When ``True``, extrapolation uses endpoint values (for ``"nearest"``
        and ``"linear"``) or cubic spline extrapolation. When ``False`` (the
        default) out-of-range targets are filled with ``fill_value``.
    spline_kwargs:
        Optional keyword arguments forwarded to
        :class:`scipy.interpolate.CubicSpline` when available.

    Returns
    -------
    Spectrum | np.ndarray
        An interpolated :class:`~alchemi.types.Spectrum` when a spectrum is
        provided as input, otherwise a ``np.ndarray`` with the same leading
        dimensions as ``source_values`` and a trailing dimension matching
        ``target_centers_nm``.
    """

    def _interpolate_values(
        wavelengths_nm: np.ndarray, values: np.ndarray, centers_nm: np.ndarray
    ) -> np.ndarray:
        check_monotonic(wavelengths_nm, strict=True)

        centers = np.asarray(centers_nm, dtype=np.float64)
        vals = np.asarray(values, dtype=np.float64)
        wl = np.asarray(wavelengths_nm, dtype=np.float64)

        if vals.shape[-1] != wl.shape[0]:
            msg = "source_values last dimension must match source_wavelengths_nm length"
            raise ValueError(msg)

        flat_vals = vals.reshape(-1, vals.shape[-1])
        result = np.empty((flat_vals.shape[0], centers.shape[0]), dtype=np.float64)

        mode_local = mode.lower()
        if mode_local not in {"nearest", "linear", "spline"}:
            raise ValueError("mode must be 'nearest', 'linear', or 'spline'")

        if mode_local == "nearest":
            idx = np.searchsorted(wl, centers, side="left")
            idx_right = np.clip(idx, 0, wl.size - 1)
            idx_left = np.clip(idx - 1, 0, wl.size - 1)

            dist_right = np.abs(wl[idx_right] - centers)
            dist_left = np.abs(wl[idx_left] - centers)
            nearest_idx = np.where(dist_left <= dist_right, idx_left, idx_right)

            for row_idx, row in enumerate(flat_vals):
                row_interp = row[nearest_idx]
                if not allow_extrapolation:
                    out_of_range = (centers < wl[0]) | (centers > wl[-1])
                    row_interp = np.where(out_of_range, fill_value, row_interp)
                result[row_idx] = row_interp

        elif mode_local == "linear":
            for row_idx, row in enumerate(flat_vals):
                left = row[0] if allow_extrapolation else fill_value
                right = row[-1] if allow_extrapolation else fill_value
                result[row_idx] = np.interp(centers, wl, row, left=left, right=right)

        else:  # spline
            try:  # pragma: no cover - optional dependency
                from scipy.interpolate import CubicSpline
            except Exception:  # pragma: no cover - optional dependency
                for row_idx, row in enumerate(flat_vals):
                    left = row[0] if allow_extrapolation else fill_value
                    right = row[-1] if allow_extrapolation else fill_value
                    result[row_idx] = np.interp(
                        centers, wl, row, left=left, right=right
                    )
            else:
                if wl.size < 4:
                    for row_idx, row in enumerate(flat_vals):
                        left = row[0] if allow_extrapolation else fill_value
                        right = row[-1] if allow_extrapolation else fill_value
                        result[row_idx] = np.interp(
                            centers, wl, row, left=left, right=right
                        )
                else:
                    kwargs = spline_kwargs or {}
                    for row_idx, row in enumerate(flat_vals):
                        spline = CubicSpline(
                            wl, row, extrapolate=allow_extrapolation, **kwargs
                        )
                        row_interp = spline(centers)
                        if not allow_extrapolation:
                            out_of_range = (centers < wl[0]) | (centers > wl[-1])
                            row_interp = np.where(out_of_range, fill_value, row_interp)
                        result[row_idx] = row_interp

        return result.reshape(*vals.shape[:-1], centers.shape[0])

    if isinstance(source_wavelengths_nm, Spectrum):
        if source_values is None and target_centers_nm is None:
            msg = "target_centers_nm must be provided when interpolating a Spectrum"
            raise ValueError(msg)
        centers = (
            np.asarray(source_values, dtype=np.float64)
            if target_centers_nm is None
            else np.asarray(target_centers_nm, dtype=np.float64)
        )
        spectrum = source_wavelengths_nm
        interpolated = _interpolate_values(spectrum.wavelengths.nm, spectrum.values, centers)
        return Spectrum(
            wavelengths=WavelengthGrid(centers),
            values=np.asarray(interpolated, dtype=np.float64),
            kind=spectrum.kind,
            units=spectrum.units,
            mask=None,
            meta=spectrum.meta.copy(),
        )

    if source_values is None or target_centers_nm is None:
        msg = "source_values and target_centers_nm must be provided for array inputs"
        raise ValueError(msg)

    return _interpolate_values(
        np.asarray(source_wavelengths_nm, dtype=np.float64),
        np.asarray(source_values, dtype=np.float64),
        np.asarray(target_centers_nm, dtype=np.float64),
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
