from __future__ import annotations

import numpy as np


def reflectance_to_radiance(
    R: np.ndarray, E0: np.ndarray, cos_sun: float, tau: float, Lpath: float
) -> np.ndarray:
    return tau * (E0 * cos_sun / np.pi) * R + Lpath


def radiance_to_reflectance(
    L: np.ndarray, E0: np.ndarray, cos_sun: float, tau: float, Lpath: float
) -> np.ndarray:
    denom = np.clip(tau * (E0 * cos_sun / np.pi), 1e-12, None)
    return np.clip((L - Lpath) / denom, 0.0, 1.5)


def continuum_remove(
    wavelength_nm: np.ndarray, reflectance: np.ndarray, left_nm: float, right_nm: float
) -> tuple[np.ndarray, np.ndarray]:
    l_idx = np.searchsorted(wavelength_nm, left_nm)
    r_idx = np.searchsorted(wavelength_nm, right_nm)
    l_ref, r_ref = reflectance[l_idx], reflectance[r_idx]
    slope = (r_ref - l_ref) / (right_nm - left_nm + 1e-12)
    cont = np.clip(l_ref + slope * (wavelength_nm - left_nm), 1e-6, None)
    return cont, reflectance / cont


def band_depth(
    wavelength_nm: np.ndarray,
    reflectance: np.ndarray,
    center_nm: float,
    left_nm: float,
    right_nm: float,
) -> float:
    _cont, removed = continuum_remove(wavelength_nm, reflectance, left_nm, right_nm)
    c_idx = np.searchsorted(wavelength_nm, center_nm)
    return float(1.0 - removed[c_idx])
