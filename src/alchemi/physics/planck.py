"""Planck-law utilities for spectral radiance and brightness temperature."""

from __future__ import annotations

import numpy as np

from alchemi.types import (
    QuantityKind,
    RadianceUnits,
    Spectrum,
    SRFMatrix,
    TemperatureUnits,
    WavelengthGrid,
)
from alchemi.utils.integrate import np_integrate

__all__ = [
    "K_B",
    "C",
    "H",
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


def _as_float64_arrays(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """Cast inputs to ``np.float64`` ndarrays without unnecessary copies."""

    return tuple(np.asarray(arr, dtype=np.float64) for arr in arrays)


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

    Tk_arr, wl_arr = _as_float64_arrays(Tk, wl_nm)
    Tk_broadcast, wl_broadcast = np.broadcast_arrays(Tk_arr, wl_arr)

    if np.any(wl_broadcast <= 0):
        raise ValueError("Wavelengths must be strictly positive.")

    Tk_safe = np.clip(Tk_broadcast, _MIN_T_K, _MAX_T_K)
    lam_m = wl_broadcast * _NM_TO_M
    prefactor = (2.0 * H * C**2) / (lam_m**5)
    exponent = (H * C) / (lam_m * K_B * Tk_safe)
    exponent = np.clip(exponent, 0.0, 700.0)

    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        expm1_term = np.expm1(exponent)

    tiny_exponent = exponent < 1e-6
    if np.any(tiny_exponent):
        # For very small exponents, use series expansion of expm1(x) ≈ x + x^2/2.
        expm1_term = np.where(
            tiny_exponent,
            exponent + 0.5 * exponent**2,
            expm1_term,
        )

    radiance_m = prefactor / expm1_term

    radiance_nm = radiance_m * _NM_TO_M

    if out is not None:
        if out.shape != radiance_nm.shape:
            raise ValueError("Output array has incorrect shape.")
        out[...] = radiance_nm
        return out

    return radiance_nm


def _band_averaged(values: np.ndarray, wavelengths: np.ndarray, srf: SRFMatrix) -> tuple[np.ndarray, np.ndarray]:
    """Convolve spectral values to sensor bands using an SRF matrix.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Band-averaged values and the SRF centers.
    """

    band_values: list[float] = []
    for nm_band, resp in zip(srf.bands_nm, srf.bands_resp, strict=True):
        nm_arr = np.asarray(nm_band, dtype=np.float64)
        resp_arr = np.asarray(resp, dtype=np.float64)
        interp_vals = np.interp(nm_arr, wavelengths, values)
        area = float(np_integrate(resp_arr, nm_arr))
        if not np.isfinite(area) or area <= 0.0:
            raise ValueError("SRF band responses must integrate to a positive value")
        numerator = float(np_integrate(interp_vals * resp_arr, nm_arr))
        band_values.append(numerator / area)

    return np.asarray(band_values, dtype=np.float64), np.asarray(srf.centers_nm, dtype=np.float64)


def _srf_centers(srf: SRFMatrix) -> np.ndarray:
    centers = np.asarray(srf.centers_nm, dtype=np.float64)
    if centers.ndim != 1:
        raise ValueError("SRF centers must be 1-D")
    if centers.size == 0:
        raise ValueError("SRF must contain at least one band")
    return centers


# ---------------------------------------------------------------------------
# Spectrum-aware wrappers
# ---------------------------------------------------------------------------

def radiance_to_bt(spectrum: Spectrum, srf: SRFMatrix | None = None) -> Spectrum:
    """Convert a radiance :class:`~alchemi.types.Spectrum` to brightness temperature.

    Parameters
    ----------
    spectrum:
        Radiance spectrum with wavelengths in nanometres and radiance units
        of W·m⁻²·sr⁻¹·nm⁻¹.
    srf:
        Optional spectral response matrix. When provided, radiance is first
        band-averaged using the SRF rows before converting each band centre to
        brightness temperature.
    """

    if spectrum.kind != QuantityKind.RADIANCE:
        raise ValueError("Input spectrum must represent radiance")

    wavelengths = np.asarray(spectrum.wavelengths.nm, dtype=np.float64)
    radiance_vals = np.asarray(spectrum.values, dtype=np.float64)

    if srf is not None:
        radiance_vals, wavelengths = _band_averaged(radiance_vals, wavelengths, srf)
        mask = srf.bad_band_mask
    else:
        mask = spectrum.mask

    bt_vals = radiance_to_bt_K(radiance_vals, wavelengths)

    return Spectrum.from_brightness_temperature(
        WavelengthGrid(wavelengths),
        bt_vals,
        units=TemperatureUnits.KELVIN,
        mask=mask,
        meta=spectrum.meta.copy(),
    )


def bt_to_radiance(spectrum: Spectrum, srf: SRFMatrix | None = None) -> Spectrum:
    """Convert brightness temperature to spectral radiance.

    When an SRF is provided, the returned radiance spectrum is defined on the
    SRF band centres to remain consistent with band-averaged BT inputs.
    """

    if spectrum.kind != QuantityKind.BRIGHTNESS_T:
        raise ValueError("Input spectrum must represent brightness temperature")

    bt_vals = np.asarray(spectrum.values, dtype=np.float64)
    wavelengths = np.asarray(spectrum.wavelengths.nm, dtype=np.float64)

    if srf is not None:
        wavelengths = _srf_centers(srf)
        if bt_vals.shape[0] != wavelengths.shape[0]:
            raise ValueError("Brightness temperature must align with SRF band centres")
        mask = srf.bad_band_mask
    else:
        mask = spectrum.mask

    radiance_vals = bt_K_to_radiance(bt_vals, wavelengths)

    return Spectrum.from_radiance(
        WavelengthGrid(wavelengths),
        radiance_vals,
        units=RadianceUnits.W_M2_SR_NM,
        mask=mask,
        meta=spectrum.meta.copy(),
    )
