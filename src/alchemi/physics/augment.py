from __future__ import annotations

from typing import Any, Union

import numpy as np

RNGLike = Union[np.random.Generator, np.random.RandomState]


def _gaussian_smoother(weights_nm: np.ndarray, sigma_nm: float) -> np.ndarray:
    """Create a Gaussian smoothing kernel for the provided wavelength grid."""

    if sigma_nm <= 0.0:
        return np.eye(weights_nm.size, dtype=float)

    wl = weights_nm.astype(float)
    distances = wl[None, :] - wl[:, None]
    kernel = np.exp(-0.5 * (distances / float(sigma_nm)) ** 2)
    kernel /= np.sum(kernel, axis=1, keepdims=True)
    return kernel


def random_swirlike_atmosphere(
    wl_nm: np.ndarray,
    rng: RNGLike,
    *,
    base_tau_range: tuple[float, float] = (0.85, 1.0),
    water_band_ranges: tuple[tuple[float, float], ...] = ((1350.0, 1450.0), (1850.0, 1950.0)),
    water_tau_drop_range: tuple[float, float] = (0.3, 0.8),
    L_path_range: tuple[float, float] = (0.0, 5.0),
    smooth_sigma_nm: float = 40.0,
    disable_water_bands: bool = False,
) -> tuple[np.ndarray, float]:
    """Sample a SWIR-like atmospheric state for a given wavelength grid."""

    wl_arr = np.asarray(wl_nm, dtype=float)
    if wl_arr.ndim != 1:
        raise ValueError("wl_nm must be a 1-D array of wavelengths")

    base_lo, base_hi = base_tau_range
    if base_lo <= 0 or base_hi <= 0:
        raise ValueError("base_tau_range must be positive")

    noise = rng.normal(0.0, 1.0, size=wl_arr.shape[0])
    kernel = _gaussian_smoother(wl_arr, smooth_sigma_nm)
    smooth_field = kernel @ noise
    smooth_field -= smooth_field.min()
    denom = smooth_field.max() - smooth_field.min()
    if denom < 1e-6:
        scaled = np.zeros_like(smooth_field)
    else:
        scaled = smooth_field / denom
    tau_vec = base_lo + (base_hi - base_lo) * scaled

    if not disable_water_bands and water_band_ranges:
        for left_nm, right_nm in water_band_ranges:
            if right_nm <= left_nm:
                continue
            drop = rng.uniform(*water_tau_drop_range)
            center = 0.5 * (left_nm + right_nm)
            width = right_nm - left_nm
            sigma = max(width / 4.0, 1e-6)
            profile = np.exp(-0.5 * ((wl_arr - center) / sigma) ** 2)
            tau_vec *= 1.0 - (1.0 - drop) * profile

    tau_vec = np.clip(tau_vec, 0.05, 1.0)

    L_path = float(rng.uniform(*L_path_range))
    return tau_vec.astype(float), L_path


def augment_radiance(
    L: np.ndarray,
    wl_nm: np.ndarray,
    rng: RNGLike,
    *,
    strength: float = 1.0,
    **atmo_kwargs: Any,
) -> np.ndarray:
    """Apply a random SWIR-like atmospheric perturbation to at-sensor radiances."""

    if strength <= 0.0:
        return np.asarray(L).copy()

    tau_vec, L_path = random_swirlike_atmosphere(wl_nm, rng, **atmo_kwargs)

    L = np.asarray(L, dtype=float)
    tau_broadcast = tau_vec.reshape((1,) * (L.ndim - 1) + tau_vec.shape)
    L_atmo = tau_broadcast * L + L_path
    L_aug = strength * L_atmo + (1.0 - strength) * L
    return np.maximum(L_aug, 0.0)


__all__ = ["augment_radiance", "random_swirlike_atmosphere"]
