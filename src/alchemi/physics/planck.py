"""Planck-law utilities for spectral radiance and brightness temperature.

This module implements wavelength-form Planck conversions, their inverses, and
helpers for band-averaged brightness temperatures consistent with the ALCHEMI
design doc (Section 5.1). Functions operate on canonical
``alchemi.spectral.Spectrum``/``Sample`` payloads and dense SRF matrices without
relying on SciPy.
"""

from __future__ import annotations

import numpy as np

from alchemi.spectral.sample import Sample
from alchemi.spectral.srf import SRFMatrix
from alchemi.types import (
    QuantityKind,
    RadianceUnits,
    Spectrum,
    TemperatureUnits,
    WavelengthGrid,
)

__all__ = [
    "K_B",
    "C",
    "H",
    "planck_radiance_wavelength",
    "inverse_planck_central_lambda",
    "band_averaged_radiance",
    "invert_band_averaged_radiance_to_bt",
    "radiance_spectrum_to_bt",
    "bt_spectrum_to_radiance",
    "radiance_sample_to_bt_sample",
    "bt_sample_to_radiance_sample",
    "bt_K_to_radiance",
    "bt_to_radiance",
    "radiance_to_bt",
    "radiance_to_bt_K",
]

# ---------------------------------------------------------------------------
# Physical constants in SI units (float64 for stability).
H: float = 6.626_070_15e-34
"""Planck constant (J·s)."""

C: float = 2.997_924_58e8
"""Speed of light in vacuum (m/s)."""

K_B: float = 1.380_649e-23
"""Boltzmann constant (J/K)."""

# Scaling helpers between nanometres and metres.
_NM_TO_M: float = 1e-9
_M_TO_NM: float = 1e9

# Numerical safeguards.
_MIN_L_M: float = 1e-50
_MIN_T_K: float = 1e-6
_MAX_T_K: float = 1e4
_MAX_RATIO: float = 1e300
_MIN_LOG_OFFSET: float = 1e-12

# Temperature brackets used for numerical inversion (Kelvin).
_DEFAULT_T_MIN: float = 150.0
_DEFAULT_T_MAX: float = 400.0
_ABS_TOL: float = 1e-6
_REL_TOL: float = 1e-8
_MAX_ITER: int = 100


def _as_float64_arrays(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """Cast inputs to ``np.float64`` ndarrays without unnecessary copies."""

    return tuple(np.asarray(arr, dtype=np.float64) for arr in arrays)


# ---------------------------------------------------------------------------
# Low-level Planck helpers
# ---------------------------------------------------------------------------

def planck_radiance_wavelength(
    wavelength_nm: np.ndarray | float,
    temperature_K: np.ndarray | float,
) -> np.ndarray:
    """Evaluate Planck's law in wavelength form.

    Parameters
    ----------
    wavelength_nm:
        Wavelength(s) in nanometres. Must be strictly positive.
    temperature_K:
        Blackbody temperature(s) in Kelvin. Must be strictly positive.

    Returns
    -------
    np.ndarray
        Spectral radiance in W·m⁻²·sr⁻¹·nm⁻¹ with shape broadcast from the
        inputs.
    """

    wl_arr, temp_arr = _as_float64_arrays(wavelength_nm, temperature_K)
    wl_b, temp_b = np.broadcast_arrays(wl_arr, temp_arr)

    if np.any(wl_b <= 0):
        raise ValueError("Wavelengths must be strictly positive")
    if np.any(temp_b <= 0):
        raise ValueError("Temperatures must be strictly positive")

    lam_m = wl_b * _NM_TO_M
    prefactor = (2.0 * H * C**2) / (lam_m**5)
    exponent = (H * C) / (lam_m * K_B * temp_b)
    exponent = np.clip(exponent, 0.0, 700.0)

    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        expm1_term = np.expm1(exponent)

    tiny_exponent = exponent < 1e-6
    if np.any(tiny_exponent):
        expm1_term = np.where(
            tiny_exponent,
            exponent + 0.5 * exponent**2,
            expm1_term,
        )

    radiance_m = prefactor / expm1_term
    return radiance_m * _NM_TO_M


def _binary_search_temperature(
    target_radiance: float,
    wavelength_nm: float,
    *,
    evaluator,
    t_min: float = _DEFAULT_T_MIN,
    t_max: float = _DEFAULT_T_MAX,
) -> float:
    """Invert a monotonic radiance evaluator using bracketed binary search."""

    if target_radiance <= 0:
        raise ValueError("Radiance must be positive for inversion")
    if wavelength_nm <= 0:
        raise ValueError("Wavelengths must be strictly positive")

    low = max(_MIN_T_K, float(t_min))
    high = max(low * 1.01, float(t_max))

    rad_low = evaluator(low)
    rad_high = evaluator(high)

    # Expand the bracket upwards if necessary.
    while target_radiance > rad_high and high < _MAX_T_K:
        low, rad_low = high, rad_high
        high = min(high * 1.5, _MAX_T_K)
        rad_high = evaluator(high)

    # Expand downward if target falls below the initial lower bound.
    while target_radiance < rad_low and low > 1.0:
        high, rad_high = low, rad_low
        low = max(1.0, low * 0.5)
        rad_low = evaluator(low)

    for _ in range(_MAX_ITER):
        mid = 0.5 * (low + high)
        rad_mid = evaluator(mid)

        if rad_mid < target_radiance:
            low = mid
        else:
            high = mid

        if abs(high - low) <= max(_ABS_TOL, _REL_TOL * high):
            break

    return 0.5 * (low + high)


def inverse_planck_central_lambda(
    radiance_W_m2_sr_nm: np.ndarray | float,
    wavelength_nm: np.ndarray | float,
    *,
    t_min: float = _DEFAULT_T_MIN,
    t_max: float = _DEFAULT_T_MAX,
) -> np.ndarray:
    """Numerically invert Planck's law at fixed wavelength(s)."""

    L_arr, wl_arr = _as_float64_arrays(radiance_W_m2_sr_nm, wavelength_nm)
    L_b, wl_b = np.broadcast_arrays(L_arr, wl_arr)

    out = np.empty_like(L_b, dtype=np.float64)
    it = np.nditer([L_b, wl_b, out], flags=["multi_index"], op_flags=[["readonly"], ["readonly"], ["writeonly"]])
    for L_val, wl_val, out_ref in it:
        evaluator = lambda temp: float(planck_radiance_wavelength(float(wl_val), temp))
        out_ref[...] = _binary_search_temperature(float(L_val), float(wl_val), evaluator=evaluator, t_min=t_min, t_max=t_max)

    return out


# ----------------------------------------------------------------------------
# Array-based Planck conversions (internal helpers)
# ----------------------------------------------------------------------------

def radiance_to_bt_K(
    L: np.ndarray,
    wl_nm: np.ndarray,
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Convert spectral radiance to brightness temperature in Kelvin.

    Parameters
    ----------
    L : np.ndarray
        Spectral radiance, in W·m^-2·sr^-1·nm^-1. Can be any shape;
        will be broadcast against `wl_nm`.
    wl_nm : np.ndarray
        Wavelengths in nanometers. Must be broadcastable to `L`.

    Returns
    -------
    bt_K : np.ndarray
        Brightness temperature in Kelvin, same shape as broadcast(L, wl_nm).

    Notes
    -----
    - Internally converts radiance from per-nm to per-m by multiplying by 1e9.
    - Uses Planck's law inversion:
        T = (h*c / (λ*k)) / ln(1 + 2*h*c**2 / (L_λ * λ**5))
      with λ in meters and L_λ in W·m^-2·sr^-1·m^-1.
    - All intermediate computations are done in float64 for numerical stability,
      even if the inputs are float16/float32.
    - Radiances that are non-positive return 0 K in the output.
    """

    L_arr, wl_arr = _as_float64_arrays(L, wl_nm)
    L_broadcast, wl_broadcast = np.broadcast_arrays(L_arr, wl_arr)

    if np.any(wl_broadcast <= 0):
        raise ValueError("Wavelengths must be strictly positive.")

    lam_m = wl_broadcast * _NM_TO_M
    L_m = np.clip(L_broadcast * _M_TO_NM, _MIN_L_M, np.inf)

    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        ratio = (2.0 * H * C**2) / (L_m * lam_m**5)
    ratio = np.clip(ratio, 0.0, _MAX_RATIO)

    arg = 1.0 + ratio
    arg = np.clip(arg, 1.0 + _MIN_LOG_OFFSET, _MAX_RATIO)
    log_term = np.log(arg)

    small_ratio = ratio < _MIN_LOG_OFFSET
    if np.any(small_ratio):
        # Use first-order expansion log(1 + x) ~ x for tiny x to avoid precision loss.
        log_term = np.where(
            small_ratio,
            np.where(ratio > 0.0, ratio, _MIN_LOG_OFFSET),
            log_term,
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        bt = ((H * C) / (lam_m * K_B)) / log_term

    bt = np.where(L_broadcast <= 0.0, 0.0, bt)

    if out is not None:
        if out.shape != bt.shape:
            raise ValueError("Output array has incorrect shape.")
        out[...] = bt
        return out

    return bt


def bt_K_to_radiance(
    Tk: np.ndarray,
    wl_nm: np.ndarray,
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Convert brightness temperature in Kelvin to spectral radiance.

    The returned radiance is expressed in W·m^-2·sr^-1·nm^-1.
    """

    radiance_nm = planck_radiance_wavelength(wl_nm, Tk)

    if out is not None:
        if out.shape != radiance_nm.shape:
            raise ValueError("Output array has incorrect shape.")
        out[...] = radiance_nm
        return out

    return radiance_nm


def band_averaged_radiance(
    temperature_K: float | np.ndarray,
    srf_matrix: np.ndarray,
    srf_wavelength_nm: np.ndarray,
) -> np.ndarray:
    """Compute band-averaged radiance for each SRF band."""

    temps = np.asarray(temperature_K, dtype=np.float64)
    srfs = np.asarray(srf_matrix, dtype=np.float64)
    wl = np.asarray(srf_wavelength_nm, dtype=np.float64)

    if wl.ndim != 1:
        raise ValueError("SRF wavelength grid must be 1-D")
    if srfs.ndim != 2:
        raise ValueError("SRF matrix must be 2-D (bands x wavelengths)")
    if srfs.shape[1] != wl.shape[0]:
        raise ValueError("SRF matrix column count must match wavelength grid length")
    if np.any(wl <= 0):
        raise ValueError("SRF wavelengths must be strictly positive")

    temps_arr = np.asarray(temps, dtype=np.float64)

    if temps_arr.ndim == 1 and temps_arr.shape[0] == srfs.shape[0]:
        return np.asarray(
            [band_averaged_radiance(temp, srfs[i : i + 1], wl)[0] for i, temp in enumerate(temps_arr)],
            dtype=np.float64,
        )

    temps_b = temps_arr[..., None]
    radiance = planck_radiance_wavelength(wl, temps_b)

    numerator = np.trapezoid(srfs * radiance[..., None, :], x=wl, axis=-1)
    denominator = np.trapezoid(srfs, x=wl, axis=-1)

    if np.any(~np.isfinite(denominator)) or np.any(denominator <= 0):
        raise ValueError("SRF rows must integrate to a positive finite area")

    averaged = numerator / denominator
    return averaged


def invert_band_averaged_radiance_to_bt(
    band_radiance_W_m2_sr_nm: np.ndarray,
    srf_matrix: np.ndarray,
    srf_wavelength_nm: np.ndarray,
    *,
    t_min: float = _DEFAULT_T_MIN,
    t_max: float = _DEFAULT_T_MAX,
) -> np.ndarray:
    """Invert band-averaged radiances to brightness temperatures."""

    band_radiance = np.asarray(band_radiance_W_m2_sr_nm, dtype=np.float64)
    srfs = np.asarray(srf_matrix, dtype=np.float64)
    wl = np.asarray(srf_wavelength_nm, dtype=np.float64)

    if wl.ndim != 1:
        raise ValueError("SRF wavelength grid must be 1-D")
    if srfs.ndim != 2:
        raise ValueError("SRF matrix must be 2-D (bands x wavelengths)")
    if srfs.shape[1] != wl.shape[0]:
        raise ValueError("SRF matrix column count must match wavelength grid length")
    if np.any(wl <= 0):
        raise ValueError("SRF wavelengths must be strictly positive")

    if band_radiance.ndim != 1:
        raise ValueError("Band radiance must be 1-D over bands")
    if band_radiance.shape[0] != srfs.shape[0]:
        raise ValueError("Band radiance length must match SRF band count")

    out = np.empty_like(band_radiance, dtype=np.float64)

    for idx, L_val in enumerate(band_radiance):
        row = srfs[idx : idx + 1, :]
        lam_eff = float(_effective_wavelengths(row, wl)[0])

        def evaluator(temp: float) -> float:
            return float(band_averaged_radiance(temp, row, wl)[0])

        out[idx] = _binary_search_temperature(L_val, lam_eff, evaluator=evaluator, t_min=t_min, t_max=t_max)

    return out


def _resolve_srf_inputs(
    srf_matrix: np.ndarray | SRFMatrix | None, srf_wavelength_nm: np.ndarray | None
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if srf_matrix is None:
        return None, srf_wavelength_nm
    if isinstance(srf_matrix, SRFMatrix):
        return srf_matrix.matrix, srf_matrix.wavelength_nm
    return np.asarray(srf_matrix, dtype=np.float64), srf_wavelength_nm


def _effective_wavelengths(srfs: np.ndarray, wl: np.ndarray) -> np.ndarray:
    numerator = np.trapezoid(srfs * wl[None, :], x=wl, axis=1)
    denominator = np.trapezoid(srfs, x=wl, axis=1)
    if np.any(~np.isfinite(denominator)) or np.any(denominator <= 0):
        raise ValueError("SRF rows must integrate to a positive finite area")
    return numerator / denominator


def radiance_spectrum_to_bt(
    spectrum: Spectrum,
    *,
    srf_matrix: np.ndarray | SRFMatrix | None = None,
    srf_wavelength_nm: np.ndarray | None = None,
    method: str = "central_lambda",
) -> Spectrum:
    """Convert a radiance spectrum to brightness temperature."""

    if spectrum.kind != QuantityKind.RADIANCE:
        raise ValueError("Input spectrum must represent radiance")

    srfs, srf_wl = _resolve_srf_inputs(srf_matrix, srf_wavelength_nm)

    radiance_vals = np.asarray(spectrum.values, dtype=np.float64)
    wavelengths = np.asarray(spectrum.wavelengths.nm, dtype=np.float64)

    if method == "central_lambda":
        if srfs is not None:
            if srf_wl is None:
                raise ValueError("srf_wavelength_nm must accompany srf_matrix")
            lambda_eff = _effective_wavelengths(np.asarray(srfs, dtype=np.float64), np.asarray(srf_wl, dtype=np.float64))
            if lambda_eff.shape[0] != radiance_vals.shape[-1]:
                raise ValueError("SRF band count must match spectrum length")
        else:
            lambda_eff = wavelengths

        bt_vals = inverse_planck_central_lambda(radiance_vals, lambda_eff)
    elif method == "band":
        if srfs is None or srf_wl is None:
            raise ValueError("Band-averaged inversion requires SRF matrix and wavelength grid")
        bt_vals = invert_band_averaged_radiance_to_bt(radiance_vals, srfs, srf_wl)
    else:
        raise ValueError("method must be 'central_lambda' or 'band'")

    return Spectrum.from_brightness_temperature(
        WavelengthGrid(wavelengths),
        bt_vals,
        units=TemperatureUnits.KELVIN,
        mask=spectrum.mask,
        meta=spectrum.meta.copy(),
    )


def bt_spectrum_to_radiance(
    spectrum_bt: Spectrum,
    *,
    srf_matrix: np.ndarray | SRFMatrix | None = None,
    srf_wavelength_nm: np.ndarray | None = None,
    method: str = "central_lambda",
) -> Spectrum:
    """Convert brightness temperatures back to spectral radiance."""

    if spectrum_bt.kind != QuantityKind.BRIGHTNESS_T:
        raise ValueError("Input spectrum must represent brightness temperature")

    srfs, srf_wl = _resolve_srf_inputs(srf_matrix, srf_wavelength_nm)

    bt_vals = np.asarray(spectrum_bt.values, dtype=np.float64)
    wavelengths = np.asarray(spectrum_bt.wavelengths.nm, dtype=np.float64)

    if method == "central_lambda":
        if srfs is not None:
            if srf_wl is None:
                raise ValueError("srf_wavelength_nm must accompany srf_matrix")
            lambda_eff = _effective_wavelengths(np.asarray(srfs, dtype=np.float64), np.asarray(srf_wl, dtype=np.float64))
            if lambda_eff.shape[0] != bt_vals.shape[-1]:
                raise ValueError("SRF band count must match spectrum length")
        else:
            lambda_eff = wavelengths
        radiance_vals = planck_radiance_wavelength(lambda_eff, bt_vals)
    elif method == "band":
        if srfs is None or srf_wl is None:
            raise ValueError("Band-averaged radiance requires SRF matrix and wavelength grid")
        radiance_vals = band_averaged_radiance(bt_vals, srfs, srf_wl)
    else:
        raise ValueError("method must be 'central_lambda' or 'band'")

    return Spectrum.from_radiance(
        WavelengthGrid(wavelengths),
        np.asarray(radiance_vals, dtype=np.float64),
        units=RadianceUnits.W_M2_SR_NM,
        mask=spectrum_bt.mask,
        meta=spectrum_bt.meta.copy(),
    )


def radiance_sample_to_bt_sample(
    sample: Sample,
    *,
    srf_matrix: np.ndarray | SRFMatrix | None = None,
    srf_wavelength_nm: np.ndarray | None = None,
    method: str = "central_lambda",
) -> Sample:
    """Convert a radiance :class:`Sample` to brightness temperature."""

    matrix = srf_matrix
    wl = srf_wavelength_nm
    if matrix is None and sample.srf_matrix is not None:
        matrix = sample.srf_matrix
        wl = sample.srf_matrix.wavelength_nm

    bt_spectrum = radiance_spectrum_to_bt(
        sample.spectrum,
        srf_matrix=matrix,
        srf_wavelength_nm=wl,
        method=method,
    )
    return Sample(
        spectrum=bt_spectrum,
        sensor_id=sample.sensor_id,
        acquisition_time=sample.acquisition_time,
        geo=sample.geo,
        viewing_geometry=sample.viewing_geometry,
        band_meta=sample.band_meta,
        srf_matrix=sample.srf_matrix,
        quality_masks=sample.quality_masks,
        ancillary=sample.ancillary,
    )


def bt_sample_to_radiance_sample(
    sample: Sample,
    *,
    srf_matrix: np.ndarray | SRFMatrix | None = None,
    srf_wavelength_nm: np.ndarray | None = None,
    method: str = "central_lambda",
) -> Sample:
    """Convert a brightness-temperature :class:`Sample` back to radiance."""

    matrix = srf_matrix
    wl = srf_wavelength_nm
    if matrix is None and sample.srf_matrix is not None:
        matrix = sample.srf_matrix
        wl = sample.srf_matrix.wavelength_nm

    radiance_spectrum = bt_spectrum_to_radiance(
        sample.spectrum,
        srf_matrix=matrix,
        srf_wavelength_nm=wl,
        method=method,
    )
    return Sample(
        spectrum=radiance_spectrum,
        sensor_id=sample.sensor_id,
        acquisition_time=sample.acquisition_time,
        geo=sample.geo,
        viewing_geometry=sample.viewing_geometry,
        band_meta=sample.band_meta,
        srf_matrix=sample.srf_matrix,
        quality_masks=sample.quality_masks,
        ancillary=sample.ancillary,
    )


# ---------------------------------------------------------------------------
# Backwards-compatible wrappers
# ---------------------------------------------------------------------------

def radiance_to_bt(spectrum: Spectrum, srf: SRFMatrix | None = None) -> Spectrum:
    return radiance_spectrum_to_bt(spectrum, srf_matrix=srf, srf_wavelength_nm=None, method="central_lambda")


def bt_to_radiance(spectrum: Spectrum, srf: SRFMatrix | None = None) -> Spectrum:
    return bt_spectrum_to_radiance(spectrum, srf_matrix=srf, srf_wavelength_nm=None, method="central_lambda")
