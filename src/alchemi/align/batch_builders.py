"""Utilities for constructing SRF-aware lab↔sensor training pairs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from alchemi.srf.avirisng import avirisng_srf_matrix
from alchemi.srf.batch_convolve import batch_convolve_lab_to_sensor
from alchemi.srf.enmap import enmap_srf_matrix
from alchemi.srf.registry import get_srf


@dataclass(slots=True)
class PairBatch:
    """Container holding matched lab and sensor spectra for a single sample."""

    lab_wavelengths_nm: np.ndarray
    lab_values: np.ndarray
    sensor_wavelengths_nm: np.ndarray
    sensor_values: np.ndarray
    sensor_id: str | None = None
    lab_id: str | None = None
    noise_cfg: dict[str, Any] | None = None
    sensor_mask: np.ndarray | None = None

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

        mask = None
        if self.sensor_mask is not None:
            mask_arr = np.asarray(self.sensor_mask, dtype=bool)
            if mask_arr.shape != sensor_wl.shape:
                msg = "sensor_mask must match sensor_wavelengths_nm shape"
                raise ValueError(msg)
            mask = mask_arr.copy()

        self.lab_wavelengths_nm = lab_wl
        self.lab_values = lab_vals
        self.sensor_wavelengths_nm = sensor_wl
        self.sensor_values = sensor_vals
        self.sensor_mask = mask


@dataclass(slots=True)
class NoiseConfig:
    """Configuration for optional Gaussian noise injection in sensor space."""

    noise_level_rel: float | Sequence[float] | np.ndarray = 0.0
    seed: int | None = None
    rng: np.random.Generator | None = None

    def generator(self) -> np.random.Generator:
        if self.rng is None:
            self.rng = np.random.default_rng(self.seed)
        return self.rng

    def levels(
        self,
        band_count: int,
        override: float | Sequence[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        raw: Any
        if override is not None:
            raw = override
        else:
            raw = self.noise_level_rel
        levels: np.ndarray = np.asarray(raw, dtype=np.float64)
        if levels.ndim == 0:
            levels = np.full((band_count,), float(levels), dtype=np.float64)
        elif levels.shape != (band_count,):
            msg = "noise levels must be scalar or match the band count"
            raise ValueError(msg)
        if np.any(levels < 0.0):
            raise ValueError("noise levels must be non-negative")
        return levels


_LabSpectrum = tuple[np.ndarray, np.ndarray]


def _normalize_lab_batch(lab_batch: Sequence[_LabSpectrum]) -> tuple[np.ndarray, np.ndarray]:
    if not lab_batch:
        raise ValueError("lab_batch must contain at least one spectrum")

    first_wl: np.ndarray = np.asarray(lab_batch[0][0], dtype=np.float64)
    if first_wl.ndim != 1 or first_wl.size < 2 or np.any(np.diff(first_wl) <= 0):
        raise ValueError("Lab wavelengths must be a strictly increasing 1-D array")

    values: list[np.ndarray] = []
    for wl, vals in lab_batch:
        wl_arr: np.ndarray = np.asarray(wl, dtype=np.float64)
        if wl_arr.shape != first_wl.shape or not np.allclose(wl_arr, first_wl):
            msg = "All lab spectra must share the same wavelength grid"
            raise ValueError(msg)
        val_arr: np.ndarray = np.asarray(vals, dtype=np.float64)
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
    override_levels: float | Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    arr: np.ndarray = np.asarray(sensor_values, dtype=np.float64)
    squeeze = False
    if arr.ndim == 1:
        arr = arr[None, :]
        squeeze = True
    elif arr.ndim != 2:
        raise ValueError("sensor_values must be a 1-D or 2-D array")

    band_count = arr.shape[-1]
    levels = cfg.levels(band_count, override=override_levels)
    if not np.any(levels > 0.0):
        return arr[0] if squeeze else arr.copy()

    sigma = np.abs(arr) * levels.reshape(1, -1)
    noise: np.ndarray = rng.normal(loc=0.0, scale=1.0, size=arr.shape)
    out: np.ndarray = arr + noise * sigma
    return out[0] if squeeze else out


def _pairs_from_projection(
    lab_batch: Sequence[_LabSpectrum],
    lab_wavelengths: np.ndarray,
    sensor_values: np.ndarray,
    sensor_wavelengths: np.ndarray,
    *,
    sensor_id: str,
    sensor_mask: np.ndarray | None = None,
    noise_levels: np.ndarray | None = None,
) -> list[PairBatch]:
    sensor_arr = np.asarray(sensor_values, dtype=np.float64)
    if sensor_arr.ndim == 1:
        sensor_arr = np.repeat(sensor_arr[None, :], len(lab_batch), axis=0)
    if sensor_arr.shape[0] != len(lab_batch):
        msg = "sensor_values batch dimension must match lab_batch length"
        raise ValueError(msg)

    noise_meta = None
    if noise_levels is not None and np.any(noise_levels > 0.0):
        noise_meta = {"noise_level_rel": np.asarray(noise_levels, dtype=np.float64).copy()}

    pairs: list[PairBatch] = []
    for idx, (_, lab_vals_row) in enumerate(lab_batch):
        pairs.append(
            PairBatch(
                lab_wavelengths_nm=lab_wavelengths,
                lab_values=np.asarray(lab_vals_row, dtype=np.float64),
                sensor_wavelengths_nm=sensor_wavelengths,
                sensor_values=sensor_arr[idx],
                sensor_id=sensor_id,
                noise_cfg=noise_meta,
                sensor_mask=sensor_mask,
            )
        )
    return pairs


def build_emit_pairs(
    lab_batch: Sequence[_LabSpectrum],
    *,
    srf: str = "emit",
    noise_cfg: NoiseConfig | None = None,
) -> list[PairBatch]:
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
    sensor_noisy = _apply_noise(sensor, cfg, rng=rng)

    sensor_wl = srf_matrix.centers_nm.astype(np.float64, copy=True)

    sensor_name = (srf_matrix.sensor or srf).lower()
    return _pairs_from_projection(
        lab_batch,
        lab_wl,
        sensor_noisy,
        sensor_wl,
        sensor_id=sensor_name,
        noise_levels=cfg.levels(sensor_wl.size),
    )


def build_enmap_pairs(
    lab_batch: Sequence[_LabSpectrum],
    *,
    cache_dir: str | Path | None = "data/srf",
    noise_level_rel_vnir: float = 0.0,
    noise_level_rel_swir: float = 0.0,
    noise_cfg: NoiseConfig | None = None,
) -> list[PairBatch]:
    """Project lab spectra to the EnMAP VNIR+SWIR grid with optional noise."""

    if len(lab_batch) == 0:
        return []

    lab_wl, lab_vals = _normalize_lab_batch(lab_batch)
    cache_root = cache_dir if cache_dir is not None else "data/srf"
    srf = enmap_srf_matrix(cache_dir=cache_root)
    sensor = batch_convolve_lab_to_sensor(lab_wl, lab_vals, srf)
    centers = np.asarray(srf.centers_nm, dtype=np.float64)
    mask = getattr(srf, "bad_band_mask", None)

    override_levels = np.where(
        centers <= 999.0,
        float(noise_level_rel_vnir),
        float(noise_level_rel_swir),
    )
    cfg = noise_cfg or NoiseConfig()
    rng = cfg.generator()
    sensor_noisy = _apply_noise(sensor, cfg, rng=rng, override_levels=override_levels)

    sensor_name = (srf.sensor or "enmap").lower()
    return _pairs_from_projection(
        lab_batch,
        lab_wl,
        sensor_noisy,
        centers,
        sensor_id=sensor_name,
        sensor_mask=mask,
        noise_levels=override_levels,
    )


def build_avirisng_pairs(
    lab_batch: Sequence[_LabSpectrum],
    *,
    cache_dir: str | Path | None = None,
    noise: float | Sequence[float] | np.ndarray | None = None,
    noise_cfg: NoiseConfig | None = None,
) -> list[PairBatch]:
    """Project lab spectra onto the AVIRIS-NG grid and optionally inject noise."""

    if len(lab_batch) == 0:
        return []

    lab_wl, lab_vals = _normalize_lab_batch(lab_batch)
    srf = avirisng_srf_matrix(cache_dir=cache_dir)
    centers = np.asarray(srf.centers_nm, dtype=np.float64)
    mask = getattr(srf, "bad_band_mask", None)

    sensor = batch_convolve_lab_to_sensor(lab_wl, lab_vals, srf)
    cfg = noise_cfg or NoiseConfig()
    rng = cfg.generator()
    sensor_noisy = _apply_noise(sensor, cfg, rng=rng, override_levels=noise)

    sensor_name = (srf.sensor or "avirisng").lower()
    return _pairs_from_projection(
        lab_batch,
        lab_wl,
        sensor_noisy,
        centers,
        sensor_id=sensor_name,
        sensor_mask=mask,
        noise_levels=cfg.levels(centers.size, override=noise),
    )


# Backwards compatibility alias retained for older tests/configs.
build_emits_pairs = build_emit_pairs


__all__ = [
    "NoiseConfig",
    "PairBatch",
    "build_avirisng_pairs",
    "build_emit_pairs",
    "build_emits_pairs",
    "build_enmap_pairs",
]
