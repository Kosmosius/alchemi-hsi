"""Spectral resampling utilities (convolution, interpolation, virtual sensors)."""

from __future__ import annotations

from dataclasses import dataclass
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
    "SyntheticSensorConfig",
    "simulate_virtual_sensor",
]


@dataclass(slots=True)
class SyntheticSensorConfig:
    """Configuration for generating reproducible virtual sensors.

    The configuration samples :attr:`n_bands` Gaussian bandpasses within
    :attr:`spectral_range_nm`, drawing full-width-half-max values between
    :attr:`min_fwhm_nm` and :attr:`max_fwhm_nm`. All random draws use
    :attr:`seed` to ensure reproducibility for training-time SRF
    randomisation and lab–overhead alignment robustness tests.
    """

    spectral_range_nm: tuple[float, float]
    n_bands: int
    min_fwhm_nm: float
    max_fwhm_nm: float
    seed: int | None = None

    def validate(self) -> None:
        start, end = self.spectral_range_nm
        if not np.isfinite(start) or not np.isfinite(end):
            raise ValueError("spectral_range_nm bounds must be finite")
        if end <= start:
            raise ValueError("spectral_range_nm upper bound must exceed lower bound")
        if self.n_bands <= 0:
            raise ValueError("n_bands must be positive")
        if self.min_fwhm_nm <= 0 or self.max_fwhm_nm <= 0:
            raise ValueError("FWHM bounds must be positive")
        if self.max_fwhm_nm < self.min_fwhm_nm:
            raise ValueError("max_fwhm_nm must be >= min_fwhm_nm")

    def rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)


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


def _sample_band_metadata(
    cfg: SyntheticSensorConfig, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    cfg.validate()

    start, end = cfg.spectral_range_nm
    centers = rng.uniform(start, end, size=cfg.n_bands)
    centers.sort()

    if cfg.min_fwhm_nm == cfg.max_fwhm_nm:
        widths = np.full(cfg.n_bands, cfg.min_fwhm_nm, dtype=np.float64)
    else:
        widths = rng.uniform(cfg.min_fwhm_nm, cfg.max_fwhm_nm, size=cfg.n_bands)

    return centers, widths


def _apply_perturbations(
    centers_nm: np.ndarray,
    widths_nm: np.ndarray,
    perturb_centers_nm: float | tuple[float, float],
    perturb_width_factor: float | tuple[float, float],
    spectral_range_nm: tuple[float, float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    def _range_to_bounds(value: float | tuple[float, float]) -> tuple[float, float]:
        if isinstance(value, tuple):
            if len(value) != 2:
                raise ValueError("Perturbation tuple must have exactly two entries")
            lo, hi = float(value[0]), float(value[1])
        else:
            mag = float(value)
            if mag < 0:
                raise ValueError("Perturbation magnitudes must be non-negative")
            lo, hi = -mag, mag
        if hi < lo:
            raise ValueError("Perturbation upper bound must be >= lower bound")
        return lo, hi

    center_lo, center_hi = _range_to_bounds(perturb_centers_nm)
    width_lo, width_hi = _range_to_bounds(perturb_width_factor)

    if center_lo == center_hi == 0.0:
        perturbed_centers = centers_nm.copy()
    else:
        shifts = rng.uniform(center_lo, center_hi, size=centers_nm.shape)
        perturbed_centers = centers_nm + shifts
        np.clip(perturbed_centers, spectral_range_nm[0], spectral_range_nm[1], out=perturbed_centers)

    if width_lo == width_hi == 0.0:
        perturbed_widths = widths_nm.copy()
    else:
        factors = 1.0 + rng.uniform(width_lo, width_hi, size=widths_nm.shape)
        perturbed_widths = widths_nm * factors

    perturbed_widths = np.clip(perturbed_widths, np.finfo(np.float64).eps, None)
    return perturbed_centers, perturbed_widths


def _gaussian_rows(
    centers_nm: np.ndarray, widths_nm: np.ndarray, wavelengths_nm: np.ndarray
) -> np.ndarray:
    sigma = widths_nm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma = sigma[:, np.newaxis]
    shifted = wavelengths_nm[np.newaxis, :] - centers_nm[:, np.newaxis]
    resp = np.exp(-0.5 * (shifted / sigma) ** 2)

    normalized = np.empty_like(resp)
    for idx, row in enumerate(resp):
        normalized[idx] = _normalize_response(row, wavelengths_nm)
    return normalized


def simulate_virtual_sensor(
    lab_wavelengths_nm: np.ndarray,
    lab_spectra: np.ndarray,
    cfg: SyntheticSensorConfig,
    perturb_centers_nm: float | tuple[float, float] = 0.0,
    perturb_width_factor: float | tuple[float, float] = 0.0,
) -> tuple[np.ndarray, DenseSRFMatrix, np.ndarray]:
    """Sample a virtual Gaussian sensor and project lab spectra onto it.

    This helper underpins SRF robustness experiments, lab–overhead alignment
    sensitivity checks, and sensor randomisation during training. Band centres
    and widths are drawn from the configuration, optionally perturbed to mimic
    calibration drifts before constructing Gaussian SRFs on the lab
    wavelength grid. The resulting SRFs preserve flat spectra (unit-area rows)
    so a constant lab spectrum remains constant after convolution.
    """

    wavelengths = np.asarray(lab_wavelengths_nm, dtype=np.float64)
    check_monotonic(wavelengths, strict=True)

    spectra = np.asarray(lab_spectra, dtype=np.float64)
    if spectra.shape[-1] != wavelengths.shape[0]:
        raise ValueError("lab_spectra last dimension must match lab_wavelengths_nm length")

    rng = cfg.rng()
    centers, widths = _sample_band_metadata(cfg, rng)
    centers, widths = _apply_perturbations(
        centers, widths, perturb_centers_nm, perturb_width_factor, cfg.spectral_range_nm, rng
    )

    srfs = _gaussian_rows(centers, widths, wavelengths)
    dense_srf = DenseSRFMatrix(wavelength_nm=wavelengths, matrix=srfs)

    band_spectra = convolve_to_bands_batched(spectra, wavelengths, dense_srf)
    return centers, dense_srf, band_spectra
