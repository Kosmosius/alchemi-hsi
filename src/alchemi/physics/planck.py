"""Planck-law utilities for spectral radiance and brightness temperature."""

from __future__ import annotations

import numpy as np

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


# Backwards compatibility aliases -------------------------------------------------


def radiance_to_bt(
    L: np.ndarray,
    wavelength_nm: np.ndarray,
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Alias for :func:`radiance_to_bt_K` for backwards compatibility."""

    return radiance_to_bt_K(L, wavelength_nm, out=out)


def bt_to_radiance(
    bt_K: np.ndarray,
    wavelength_nm: np.ndarray,
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Alias for :func:`bt_K_to_radiance` for backwards compatibility."""

    return bt_K_to_radiance(bt_K, wavelength_nm, out=out)
