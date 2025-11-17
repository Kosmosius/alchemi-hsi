"""Utilities for constructing lab/sensor training pairs for alignment."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..srf.batch_convolve import batch_convolve_lab_to_sensor
from ..srf.registry import get_srf


@dataclass(slots=True)
class Pair:
    """Container holding matched lab and sensor spectra."""

    lab_wavelengths_nm: np.ndarray
    lab_values: np.ndarray
    sensor_wavelengths_nm: np.ndarray
    sensor_values: np.ndarray

    def __post_init__(self) -> None:
        lab_wl = np.asarray(self.lab_wavelengths_nm, dtype=np.float64).copy()
        if lab_wl.ndim != 1 or lab_wl.size < 2 or np.any(np.diff(lab_wl) <= 0):
            msg = "lab_wavelengths_nm must be a strictly increasing 1-D array"
            raise ValueError(msg)
        lab_vals = np.asarray(self.lab_values, dtype=np.float64).copy()
        if lab_vals.ndim != 1 or lab_vals.shape[0] != lab_wl.shape[0]:
            msg = "lab_values must be a 1-D array matching lab wavelengths"
            raise ValueError(msg)

        sensor_wl = np.asarray(self.sensor_wavelengths_nm, dtype=np.float64).copy()
        if sensor_wl.ndim != 1 or sensor_wl.size < 1 or np.any(np.diff(sensor_wl) <= 0):
            msg = "sensor_wavelengths_nm must be a strictly increasing 1-D array"
            raise ValueError(msg)
        sensor_vals = np.asarray(self.sensor_values, dtype=np.float64).copy()
        if sensor_vals.ndim != 1 or sensor_vals.shape[0] != sensor_wl.shape[0]:
            msg = "sensor_values must be a 1-D array matching sensor wavelengths"
            raise ValueError(msg)

        self.lab_wavelengths_nm = lab_wl
        self.lab_values = lab_vals
        self.sensor_wavelengths_nm = sensor_wl
        self.sensor_values = sensor_vals


@dataclass(slots=True)
class NoiseConfig:
    """Configuration for optional Gaussian noise injection in sensor space."""

    noise_level_rel: float = 0.0
    seed: int | None = None
    rng: np.random.Generator | None = None

    def __post_init__(self) -> None:
        if self.noise_level_rel < 0.0:
            msg = "noise_level_rel must be non-negative"
            raise ValueError(msg)

    def generator(self) -> np.random.Generator:
        if self.rng is None:
            self.rng = np.random.default_rng(self.seed)
        return self.rng

    @property
    def enabled(self) -> bool:
        return self.noise_level_rel > 0.0


_LabSpectrum = tuple[np.ndarray, np.ndarray]


def _normalize_lab_batch(lab_batch: Sequence[_LabSpectrum]) -> tuple[np.ndarray, np.ndarray]:
    if not lab_batch:
        raise ValueError("lab_batch must contain at least one spectrum")

    first_wl = np.asarray(lab_batch[0][0], dtype=np.float64)
    if first_wl.ndim != 1 or first_wl.size < 2 or np.any(np.diff(first_wl) <= 0):
        raise ValueError("Lab wavelengths must be a strictly increasing 1-D array")

    values: list[np.ndarray] = []
    for wl, vals in lab_batch:
        wl_arr = np.asarray(wl, dtype=np.float64)
        if wl_arr.shape != first_wl.shape or not np.allclose(wl_arr, first_wl):
            msg = "All lab spectra must share the same wavelength grid"
            raise ValueError(msg)
        val_arr = np.asarray(vals, dtype=np.float64)
        if val_arr.ndim != 1 or val_arr.shape[0] != wl_arr.shape[0]:
            msg = "Lab spectrum values must match wavelength grid length"
            raise ValueError(msg)
        values.append(val_arr)

    return first_wl, np.stack(values, axis=0)


def _apply_noise(
    sensor_values: np.ndarray,
    cfg: NoiseConfig,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    sigma = np.abs(sensor_values) * cfg.noise_level_rel
    if not np.any(sigma > 0.0):
        return sensor_values.copy()
    noise = rng.normal(loc=0.0, scale=1.0, size=sensor_values.shape)
    return sensor_values + noise * sigma


def build_emits_pairs(
    lab_batch: Sequence[_LabSpectrum],
    *,
    srf: str = "emit",
    noise_cfg: NoiseConfig | None = None,
) -> list[Pair]:
    """Project high-resolution lab spectra onto EMIT bands and bundle pairs."""

    if len(lab_batch) == 0:
        return []

    lab_wl, lab_vals = _normalize_lab_batch(lab_batch)
    srf_matrix, _ = get_srf(srf, wavelengths_nm=lab_wl)
    sensor = batch_convolve_lab_to_sensor(lab_wl, lab_vals, srf_matrix)
    if sensor.ndim == 1:
        sensor = sensor[None, :]

    cfg = noise_cfg or NoiseConfig()
    rng = cfg.generator()
    if cfg.enabled:
        sensor_noisy = _apply_noise(sensor, cfg, rng=rng)
    else:
        sensor_noisy = sensor.copy()

    sensor_wl = srf_matrix.centers_nm.astype(np.float64, copy=True)

    pairs: list[Pair] = []
    for idx, (_, lab_vals_row) in enumerate(lab_batch):
        pairs.append(
            Pair(
                lab_wavelengths_nm=lab_wl,
                lab_values=np.asarray(lab_vals_row, dtype=np.float64),
                sensor_wavelengths_nm=sensor_wl,
                sensor_values=sensor_noisy[idx],
            )
        )
    return pairs


__all__ = ["NoiseConfig", "Pair", "build_emits_pairs"]
