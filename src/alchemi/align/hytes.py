"""Batch builders for aligning laboratory spectra with HyTES observations."""

from __future__ import annotations

# mypy: ignore-errors
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass

import numpy as np

from alchemi.physics import planck
from alchemi.srf.batch_convolve import batch_convolve_lab_to_sensor
from alchemi.srf.hytes import hytes_srf_matrix

_NM_TO_M = 1e-9
_MIN_T_K = 1e-6
_MAX_T_K = 1e4
_MAX_EXPONENT = 700.0
_MIN_EXPM1 = 1e-300


@dataclass
class HyTESNoiseConfig:
    """Configuration for injecting HyTES-like brightness-temperature noise."""

    enabled: bool = False
    nedt_mK: float = 200.0
    rng: np.random.Generator | None = None

    @property
    def std_K(self) -> float:
        """Return the Gaussian standard deviation in Kelvin."""

        return float(self.nedt_mK) * 1e-3


def build_hytes_pairs(
    lab_batch: Mapping[str, np.ndarray], *, noise_cfg: HyTESNoiseConfig | None = None
) -> MutableMapping[str, np.ndarray]:
    """Project laboratory radiance onto HyTES bands with optional BT noise."""

    if noise_cfg is None:
        noise_cfg = HyTESNoiseConfig()

    lab_nm = np.asarray(lab_batch["wavelengths_nm"], dtype=np.float64)
    lab_values = np.asarray(lab_batch["radiance"], dtype=np.float64)
    if lab_values.ndim == 1:
        lab_values = lab_values[None, :]
    if lab_nm.ndim != 1:
        raise ValueError("wavelengths_nm must be 1-D")
    if lab_values.shape[-1] != lab_nm.shape[0]:
        raise ValueError("radiance length must match wavelength grid")

    srf = hytes_srf_matrix()
    radiance_clean = batch_convolve_lab_to_sensor(lab_nm, lab_values, srf)
    hytes_nm = np.asarray(srf.centers_nm, dtype=np.float64)

    bt = planck.radiance_to_bt_K(radiance_clean, hytes_nm)
    bt_noise = np.zeros_like(bt)
    radiance = radiance_clean.copy()

    if noise_cfg.enabled and noise_cfg.std_K > 0.0:
        rng = noise_cfg.rng or np.random.default_rng()
        bt_noise = rng.normal(loc=0.0, scale=noise_cfg.std_K, size=bt.shape)
        dL_dT = _planck_radiance_derivative(bt, hytes_nm)
        radiance = radiance + dL_dT * bt_noise

    result: MutableMapping[str, np.ndarray] = {
        "wavelengths_nm": hytes_nm.copy(),
        "radiance": radiance,
        "radiance_clean": radiance_clean,
        "bt": bt,
        "bt_noise": bt_noise,
    }
    return result


def _planck_radiance_derivative(bt_K: np.ndarray, wl_nm: np.ndarray) -> np.ndarray:
    """Return ∂L/∂T for Planck radiance at ``bt_K`` and wavelengths ``wl_nm``."""

    Tk = np.asarray(bt_K, dtype=np.float64)
    wl = np.asarray(wl_nm, dtype=np.float64)
    if wl.ndim != 1:
        raise ValueError("HyTES wavelengths must be 1-D")

    Tk_safe = np.clip(Tk, _MIN_T_K, _MAX_T_K)
    reshape = (1,) * (Tk_safe.ndim - 1) + (wl.shape[0],)
    wl_broadcast = np.broadcast_to(wl.reshape(reshape), Tk_safe.shape)
    lam_m = wl_broadcast * _NM_TO_M

    prefactor = (2.0 * planck.H * planck.C**2) / np.power(lam_m, 5.0)
    exponent = (planck.H * planck.C) / (lam_m * planck.K_B * Tk_safe)
    exponent = np.clip(exponent, 0.0, _MAX_EXPONENT)

    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        exp_term = np.exp(exponent)
        expm1_term = np.expm1(exponent)

    expm1_term = np.clip(expm1_term, _MIN_EXPM1, np.inf)

    derivative_m = prefactor * (exponent / Tk_safe) * exp_term / (expm1_term**2)
    derivative_nm = derivative_m * _NM_TO_M
    return derivative_nm


__all__ = ["HyTESNoiseConfig", "build_hytes_pairs"]
