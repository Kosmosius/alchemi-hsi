"""Planck-law utilities for spectral radiance and brightness temperature.

This module implements wavelength-form Planck conversions, their inverses, and
helpers for band-averaged brightness temperatures consistent with the ALCHEMI
design doc (Section 5.1). Functions operate on canonical
``alchemi.spectral.Spectrum``/``Sample`` payloads and dense SRF matrices without
relying on SciPy.
"""

from __future__ import annotations

import warnings

import numpy as np

from alchemi.spectral.sample import Sample
from alchemi.spectral.srf import SRFMatrix
from alchemi.srf.hytes import hytes_srf_matrix
from alchemi.srf.registry import sensor_srf_from_legacy
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
    "hytes_band_averaged_radiance_to_bt",
    "radiance_spectrum_to_bt",
    "radiance_spectrum_to_bt_central",
    "radiance_spectrum_to_bt_band",
    "bt_spectrum_to_radiance",
    "bt_spectrum_to_radiance_central",
    "bt_spectrum_to_radiance_band",
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
    it = np.nditer(
        [L_b, wl_b, out],
        flags=["multi_index"],
        op_flags=[["readonly"], ["readonly"], ["writeonly"]],
    )
    for L_val, wl_val, out_ref in it:
        evaluator = lambda temp, wl_val=wl_val: float(
            planck_radiance_wavelength(float(wl_val), temp)
        )
        out_ref[...] = _binary_search_temperature(
            float(L_val), float(wl_val), evaluator=evaluator, t_min=t_min, t_max=t_max
        )

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
    """Compute band-averaged radiance for each SRF band.

    Parameters
    ----------
    temperature_K:
        Blackbody temperature(s) in Kelvin. Can be scalar or array with any
        leading shape; a trailing band dimension is *not* required.
    srf_matrix:
        Spectral response functions of shape ``(bands, wavelengths)``.
    srf_wavelength_nm:
        Wavelength grid, in nanometres, corresponding to the SRF columns.

    Returns
    -------
    np.ndarray
        Band-averaged radiance in W·m⁻²·sr⁻¹·nm⁻¹ with shape
        ``temperature_K.shape + (bands,)``.
    """

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

    if temps_arr.ndim > 0 and temps_arr.shape[-1] == srfs.shape[0]:
        temps_b = temps_arr[..., :, None]
        radiance = planck_radiance_wavelength(wl, temps_b)
        numerator = np.trapezoid(radiance * srfs[None, ...], x=wl, axis=-1)
        denominator = np.trapezoid(srfs, x=wl, axis=-1)

        if np.any(~np.isfinite(denominator)) or np.any(denominator <= 0):
            raise ValueError("SRF rows must integrate to a positive finite area")

        return numerator / denominator

    temps_b = temps_arr[..., None]
    radiance = planck_radiance_wavelength(wl, temps_b)

    radiance_flat = radiance.reshape(-1, wl.shape[0])
    numerator_flat = np.trapezoid(
        radiance_flat[:, None, :] * srfs[None, ...], x=wl, axis=2
    )
    denominator = np.trapezoid(srfs, x=wl, axis=-1)

    if np.any(~np.isfinite(denominator)) or np.any(denominator <= 0):
        raise ValueError("SRF rows must integrate to a positive finite area")

    averaged_flat = numerator_flat / denominator
    output_shape = radiance.shape[:-1] + (srfs.shape[0],)
    return averaged_flat.reshape(output_shape)


def invert_band_averaged_radiance_to_bt(
    band_radiance_W_m2_sr_nm: np.ndarray,
    srf_matrix: np.ndarray,
    srf_wavelength_nm: np.ndarray,
    *,
    t_min: float = _DEFAULT_T_MIN,
    t_max: float = _DEFAULT_T_MAX,
    temps_grid_K: np.ndarray | None = None,
    strict: bool = False,
) -> np.ndarray:
    """Invert band-averaged radiances to brightness temperatures.

    Parameters
    ----------
    band_radiance_W_m2_sr_nm:
        Band-averaged radiances expressed in W·m⁻²·sr⁻¹·nm⁻¹. The last
        dimension must match the number of SRF bands; any leading dimensions are
        treated as pixels and are preserved in the output.
    srf_matrix:
        Spectral response functions with shape ``(bands, wavelengths)``. Each
        row is expected to integrate to a positive, finite value.
    srf_wavelength_nm:
        Wavelength grid, in nanometres, corresponding to the SRF columns.
    temps_grid_K:
        Optional 1-D array of temperatures in Kelvin used to tabulate the
        band-averaged radiance look-up. If omitted a default grid spanning
        200–350 K in 0.25 K steps is used. Values must be strictly positive and
        strictly increasing.
    strict:
        If ``True``, radiances that fall outside the look-up table range raise a
        ``ValueError``. If ``False`` (default) the inversion saturates to the
        nearest table temperature and emits a warning.

    Returns
    -------
    np.ndarray
        Brightness temperatures in Kelvin with the same shape as
        ``band_radiance_W_m2_sr_nm``.
    """

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

    if t_max <= t_min:
        raise ValueError("t_max must be greater than t_min")

    temps_grid = temps_grid_K
    if temps_grid is None:
        temps_grid = np.arange(float(t_min), float(t_max) + 0.25, 0.25, dtype=np.float64)
    else:
        temps_grid = np.asarray(temps_grid, dtype=np.float64)

    if temps_grid.ndim != 1:
        raise ValueError("temps_grid_K must be one-dimensional")
    if temps_grid.size < 2:
        raise ValueError("temps_grid_K must contain at least two points")
    if np.any(temps_grid <= 0):
        raise ValueError("temps_grid_K must be strictly positive")
    if not np.all(np.diff(temps_grid) > 0):
        raise ValueError("temps_grid_K must be strictly increasing")

    band_count = srfs.shape[0]
    flat = band_radiance.reshape(-1, band_count)
    if flat.shape[1] != band_count:
        raise ValueError("Band radiance last dimension must match SRF band count")

    table = band_averaged_radiance(temps_grid, srfs, wl)
    if table.shape != (temps_grid.size, band_count):
        raise ValueError("Band-averaged radiance table has unexpected shape")

    if np.any(np.diff(table, axis=0) <= 0):
        raise ValueError("Band-averaged radiance must be monotone increasing in temperature")

    temps_out = np.empty_like(flat)
    for band_idx in range(band_count):
        xp = table[:, band_idx]
        temps_out[:, band_idx] = np.interp(
            flat[:, band_idx],
            xp,
            temps_grid,
            left=temps_grid[0],
            right=temps_grid[-1],
        )

    out_of_bounds_low = flat < table[0]
    out_of_bounds_high = flat > table[-1]
    if (out_of_bounds_low | out_of_bounds_high).any():
        message = (
            "Band radiance outside inversion grid; temperatures saturated to "
            f"[{temps_grid[0]:.2f}, {temps_grid[-1]:.2f}] K"
        )
        if strict:
            raise ValueError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    temps_out = temps_out.reshape(band_radiance.shape)
    return temps_out


def hytes_band_averaged_radiance_to_bt(
    band_radiance_W_m2_sr_nm: np.ndarray,
    *,
    sensor_srf: SRFMatrix | np.ndarray | None = None,
    temps_grid_K: np.ndarray | None = None,
    strict: bool = False,
    grid_points: int = 2048,
) -> np.ndarray:
    """Convenience wrapper for HyTES-style band-averaged inversion."""

    if isinstance(sensor_srf, SRFMatrix):
        dense = sensor_srf
    else:
        srf = sensor_srf if sensor_srf is not None else hytes_srf_matrix()
        grid = np.linspace(float(srf.bands_nm[0][0]), float(srf.bands_nm[-1][-1]), grid_points, dtype=np.float64)
        dense = sensor_srf_from_legacy(srf, grid=grid).as_matrix()
        dense.normalize_rows_trapz()
    return invert_band_averaged_radiance_to_bt(
        band_radiance_W_m2_sr_nm,
        dense.matrix,
        dense.wavelength_nm,
        temps_grid_K=temps_grid_K,
        strict=strict,
    )


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
    temps_grid_K: np.ndarray | None = None,
    strict: bool = False,
) -> Spectrum:
    """Convert a radiance spectrum to brightness temperature.

    Two approximations are available via ``method``:

    - ``"central_lambda"`` (default): uses the SRF's effective band centre
      (λ_eff) or the spectrum's native wavelength grid if no SRF is provided.
      This is the recommended choice for narrow bands or when SRFs are
      unavailable.
    - ``"band"``: performs full SRF-weighted band averaging using
      ``srf_matrix``/``srf_wavelength_nm`` and numerically inverts the result.
      Prefer this for HyTES-like or other broad, non-trivial SRFs when accurate
      band-averaged BTs are required.

    ``"band"`` requires a valid SRF matrix aligned to ``srf_wavelength_nm`` and
    optionally accepts ``temps_grid_K`` to tune the inversion grid.
    """

    if spectrum.kind != QuantityKind.RADIANCE:
        raise ValueError("Input spectrum must represent radiance")

    srfs, srf_wl = _resolve_srf_inputs(srf_matrix, srf_wavelength_nm)

    radiance_vals = np.asarray(spectrum.values, dtype=np.float64)
    wavelengths = np.asarray(spectrum.wavelengths.nm, dtype=np.float64)

    if method == "central_lambda":
        if srfs is not None:
            if srf_wl is None:
                raise ValueError("srf_wavelength_nm must accompany srf_matrix")
            lambda_eff = _effective_wavelengths(
                np.asarray(srfs, dtype=np.float64), np.asarray(srf_wl, dtype=np.float64)
            )
            if lambda_eff.shape[0] != radiance_vals.shape[-1]:
                raise ValueError("SRF band count must match spectrum length")
        else:
            lambda_eff = wavelengths

        bt_vals = inverse_planck_central_lambda(radiance_vals, lambda_eff)
    elif method == "band":
        if srfs is None or srf_wl is None:
            raise ValueError("Band-averaged inversion requires SRF matrix and wavelength grid")
        bt_vals = invert_band_averaged_radiance_to_bt(
            radiance_vals, srfs, srf_wl, temps_grid_K=temps_grid_K, strict=strict
        )
    else:
        raise ValueError("method must be 'central_lambda' or 'band'")

    return Spectrum.from_brightness_temperature(
        WavelengthGrid(wavelengths),
        bt_vals,
        units=TemperatureUnits.KELVIN,
        mask=spectrum.mask,
        meta=spectrum.meta.copy(),
    )


def radiance_spectrum_to_bt_central(
    spectrum: Spectrum,
    *,
    srf_matrix: np.ndarray | SRFMatrix | None = None,
    srf_wavelength_nm: np.ndarray | None = None,
) -> Spectrum:
    """Brightness temperature using the central-wavelength approximation.

    This explicitly selects ``method="central_lambda"`` in
    :func:`radiance_spectrum_to_bt`, which applies Planck inversion at λ_eff.
    Use for narrow bands or when SRF details are not available.
    """

    return radiance_spectrum_to_bt(
        spectrum,
        srf_matrix=srf_matrix,
        srf_wavelength_nm=srf_wavelength_nm,
        method="central_lambda",
    )


def radiance_spectrum_to_bt_band(
    spectrum: Spectrum,
    *,
    srf_matrix: np.ndarray | SRFMatrix,
    srf_wavelength_nm: np.ndarray,
    temps_grid_K: np.ndarray | None = None,
    strict: bool = False,
) -> Spectrum:
    """Brightness temperature using SRF-based band averaging.

    This explicitly selects ``method="band"`` in :func:`radiance_spectrum_to_bt`
    and requires an SRF matrix aligned to ``srf_wavelength_nm``. Recommended for
    wide or structured SRFs (e.g., HyTES) when band-averaged accuracy matters.
    """

    return radiance_spectrum_to_bt(
        spectrum,
        srf_matrix=srf_matrix,
        srf_wavelength_nm=srf_wavelength_nm,
        method="band",
        temps_grid_K=temps_grid_K,
        strict=strict,
    )


def bt_spectrum_to_radiance(
    spectrum_bt: Spectrum,
    *,
    srf_matrix: np.ndarray | SRFMatrix | None = None,
    srf_wavelength_nm: np.ndarray | None = None,
    method: str = "central_lambda",
) -> Spectrum:
    """Convert brightness temperatures back to spectral radiance.

    - ``"central_lambda"`` (default): evaluates Planck's law at λ_eff (derived
      from SRFs when provided, otherwise the spectrum wavelengths). Use for
      narrow bands or when SRFs are absent.
    - ``"band"``: returns SRF-averaged radiance per band using the provided SRF
      matrix/wavelength grid. Choose this for structured or wide SRFs when a
      true band-average is needed.

    ``"band"`` requires a valid SRF matrix aligned to ``srf_wavelength_nm``.
    """

    if spectrum_bt.kind != QuantityKind.BRIGHTNESS_T:
        raise ValueError("Input spectrum must represent brightness temperature")

    srfs, srf_wl = _resolve_srf_inputs(srf_matrix, srf_wavelength_nm)

    bt_vals = np.asarray(spectrum_bt.values, dtype=np.float64)
    wavelengths = np.asarray(spectrum_bt.wavelengths.nm, dtype=np.float64)

    if method == "central_lambda":
        if srfs is not None:
            if srf_wl is None:
                raise ValueError("srf_wavelength_nm must accompany srf_matrix")
            lambda_eff = _effective_wavelengths(
                np.asarray(srfs, dtype=np.float64), np.asarray(srf_wl, dtype=np.float64)
            )
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


def bt_spectrum_to_radiance_central(
    spectrum_bt: Spectrum,
    *,
    srf_matrix: np.ndarray | SRFMatrix | None = None,
    srf_wavelength_nm: np.ndarray | None = None,
) -> Spectrum:
    """Radiance from BTs using the central-wavelength approximation."""

    return bt_spectrum_to_radiance(
        spectrum_bt,
        srf_matrix=srf_matrix,
        srf_wavelength_nm=srf_wavelength_nm,
        method="central_lambda",
    )


def bt_spectrum_to_radiance_band(
    spectrum_bt: Spectrum,
    *,
    srf_matrix: np.ndarray | SRFMatrix,
    srf_wavelength_nm: np.ndarray,
) -> Spectrum:
    """Radiance from BTs using SRF-based band averaging."""

    return bt_spectrum_to_radiance(
        spectrum_bt,
        srf_matrix=srf_matrix,
        srf_wavelength_nm=srf_wavelength_nm,
        method="band",
    )


def radiance_sample_to_bt_sample(
    sample: Sample,
    *,
    srf_matrix: np.ndarray | SRFMatrix | None = None,
    srf_wavelength_nm: np.ndarray | None = None,
    method: str = "central_lambda",
    temps_grid_K: np.ndarray | None = None,
    strict: bool = False,
) -> Sample:
    """Convert a radiance :class:`Sample` to brightness temperature.

    The ``method`` argument mirrors :func:`radiance_spectrum_to_bt`. Use
    ``"central_lambda"`` for narrow bands or absent SRFs; use ``"band"`` with a
    valid SRF matrix/grid when SRF-weighted band averages are required.
    """

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
        temps_grid_K=temps_grid_K,
        strict=strict,
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
    """Convert a brightness-temperature :class:`Sample` back to radiance.

    The ``method`` argument mirrors :func:`bt_spectrum_to_radiance`. The
    ``"band"`` option requires SRF information to return SRF-averaged radiance.
    """

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
    """Convert radiance to BT using the central-wavelength approximation."""

    return radiance_spectrum_to_bt(
        spectrum, srf_matrix=srf, srf_wavelength_nm=None, method="central_lambda"
    )


def bt_to_radiance(spectrum: Spectrum, srf: SRFMatrix | None = None) -> Spectrum:
    """Convert BT to radiance using the central-wavelength approximation."""

    return bt_spectrum_to_radiance(
        spectrum, srf_matrix=srf, srf_wavelength_nm=None, method="central_lambda"
    )
