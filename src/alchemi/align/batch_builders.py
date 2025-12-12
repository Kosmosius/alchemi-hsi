"""Utilities for constructing SRF-aware lab↔sensor training pairs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from alchemi.registry import srfs
from alchemi.spectral import Sample, Spectrum
from alchemi.spectral.sample import BandMetadata
from alchemi.srf.registry import sensor_srf_from_legacy
from alchemi.srf.resample import resample_values_with_srf
from alchemi.srf.synthetic import SRFJitterConfig, jitter_sensor_srf
from alchemi.srf.utils import resolve_band_widths
from alchemi.types import QuantityKind, ReflectanceUnits, SRFMatrix


@dataclass(slots=True)
class PairBatch:
    """Container holding matched lab and sensor spectra for a single sample."""

    lab_spectrum: Spectrum
    sensor_sample: Sample
    lab_id: str | None = None
    noise_cfg: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.lab_spectrum = self._ensure_spectrum(self.lab_spectrum)
        self.sensor_sample = self._ensure_sample(self.sensor_sample)

    @staticmethod
    def _ensure_spectrum(spectrum: Spectrum | Any) -> Spectrum:
        if isinstance(spectrum, Spectrum):
            return spectrum
        raise TypeError("lab_spectrum must be a Spectrum instance")

    @staticmethod
    def _ensure_sample(sample: Sample | Any) -> Sample:
        if isinstance(sample, Sample):
            return sample
        raise TypeError("sensor_sample must be a Sample instance")

    @property
    def lab_wavelengths_nm(self) -> np.ndarray:
        return np.asarray(self.lab_spectrum.wavelength_nm, dtype=np.float64)

    @property
    def lab_values(self) -> np.ndarray:
        return np.asarray(self.lab_spectrum.values, dtype=np.float64).reshape(-1)

    @property
    def sensor_wavelengths_nm(self) -> np.ndarray:
        return np.asarray(self.sensor_sample.spectrum.wavelength_nm, dtype=np.float64)

    @property
    def sensor_values(self) -> np.ndarray:
        return np.asarray(self.sensor_sample.spectrum.values, dtype=np.float64).reshape(-1)

    @property
    def sensor_id(self) -> str:
        return self.sensor_sample.sensor_id

    @property
    def sensor_mask(self) -> np.ndarray | None:
        return self.sensor_sample.quality_masks.get("sensor_mask")


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
        levels = np.asarray(raw, dtype=np.float64)
        if levels.ndim == 0:
            levels = np.full((band_count,), float(levels), dtype=np.float64)
        elif levels.shape != (band_count,):
            msg = "noise levels must be scalar or match the band count"
            raise ValueError(msg)
        if np.any(levels < 0.0):
            raise ValueError("noise levels must be non-negative")
        return levels


_LabSpectrum = Spectrum | tuple[np.ndarray, np.ndarray]


def _as_spectrum(candidate: _LabSpectrum) -> Spectrum:
    if isinstance(candidate, Spectrum):
        return candidate
    wavelengths, values = candidate
    return Spectrum(
        wavelengths=wavelengths,
        values=np.asarray(values, dtype=np.float64),
        kind=QuantityKind.SURFACE_REFLECTANCE,
        units=ReflectanceUnits.FRACTION,
    )


def _normalize_lab_batch(
    lab_batch: Sequence[_LabSpectrum],
) -> tuple[np.ndarray, np.ndarray, list[Spectrum]]:
    if not lab_batch:
        raise ValueError("lab_batch must contain at least one spectrum")

    spectra = [_as_spectrum(spec) for spec in lab_batch]
    first_wl = spectra[0].wavelength_nm
    if first_wl.ndim != 1 or first_wl.size < 2 or np.any(np.diff(first_wl) <= 0):
        raise ValueError("Lab wavelengths must be a strictly increasing 1-D array")

    values: list[np.ndarray] = []
    for spec in spectra:
        wl_arr = np.asarray(spec.wavelength_nm, dtype=np.float64)
        if wl_arr.shape != first_wl.shape or not np.allclose(wl_arr, first_wl):
            msg = "All lab spectra must share the same wavelength grid"
            raise ValueError(msg)
        val_arr = np.asarray(spec.values, dtype=np.float64)
        if val_arr.ndim != 1 or val_arr.shape[0] != wl_arr.shape[0]:
            msg = "Lab spectrum values must match wavelength grid length"
            raise ValueError(msg)
        values.append(val_arr)

    return first_wl, np.stack(values, axis=0), spectra


def _apply_noise(
    sensor_values: np.ndarray,
    cfg: NoiseConfig,
    *,
    rng: np.random.Generator,
    override_levels: float | Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    arr = np.asarray(sensor_values, dtype=np.float64)
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
    noise = rng.normal(loc=0.0, scale=1.0, size=arr.shape)
    out = arr + noise * sigma
    return out[0] if squeeze else out


def _maybe_jitter_srf(
    sensor_srf: Any, jitter_cfg: SRFJitterConfig | None
) -> Any:
    if jitter_cfg is None or not jitter_cfg.enabled:
        return sensor_srf

    canonical = sensor_srf
    if isinstance(sensor_srf, SRFMatrix):
        canonical = sensor_srf_from_legacy(sensor_srf)
    if not hasattr(canonical, "band_centers_nm"):
        return sensor_srf

    return jitter_sensor_srf(canonical, jitter_cfg, rng=jitter_cfg.generator())


def _ensure_sensor_srf(sensor_srf: Any) -> Any:
    if isinstance(sensor_srf, SRFMatrix):
        return sensor_srf_from_legacy(sensor_srf)
    return sensor_srf


def _sensor_identifier(sensor_srf: Any) -> str:
    if hasattr(sensor_srf, "sensor_id"):
        return str(getattr(sensor_srf, "sensor_id"))
    if hasattr(sensor_srf, "sensor"):
        return str(getattr(sensor_srf, "sensor"))
    return "unknown"


def _pairs_from_projection(
    lab_batch: Sequence[_LabSpectrum],
    lab_wavelengths: np.ndarray,
    sensor_values: np.ndarray,
    sensor_wavelengths: np.ndarray,
    *,
    sensor_id: str,
    sensor_mask: np.ndarray | None = None,
    noise_levels: np.ndarray | None = None,
    band_widths_nm: np.ndarray | None = None,
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

    lab_spectra = [_as_spectrum(item) for item in lab_batch]
    sensor_wl = np.asarray(sensor_wavelengths, dtype=np.float64)
    if sensor_mask is not None:
        mask_arr = np.asarray(sensor_mask, dtype=bool)
        if mask_arr.shape != sensor_wl.shape:
            msg = "sensor_mask must match sensor_wavelengths_nm shape"
            raise ValueError(msg)
        valid_mask = mask_arr
    else:
        valid_mask = np.ones_like(sensor_wl, dtype=bool)

    if band_widths_nm is None:
        widths, width_from_default, _ = resolve_band_widths(sensor_id, sensor_wl)
    else:
        widths = np.asarray(band_widths_nm, dtype=np.float64)
        width_from_default = np.zeros_like(widths, dtype=bool)
    band_meta = BandMetadata(
        center_nm=sensor_wl,
        width_nm=widths,
        valid_mask=valid_mask,
        srf_source=np.full(sensor_wl.shape, sensor_id),
        srf_provenance=np.full(sensor_wl.shape, "none", dtype=object),
        srf_approximate=np.full(sensor_wl.shape, True, dtype=bool),
        width_from_default=width_from_default,
    )

    pairs: list[PairBatch] = []
    for idx, lab_spec in enumerate(lab_spectra):
        sensor_spectrum = Spectrum(
            wavelengths=sensor_wl,
            values=sensor_arr[idx],
            kind=QuantityKind.SURFACE_REFLECTANCE,
            units=ReflectanceUnits.FRACTION,
            mask=None if sensor_mask is None else np.asarray(sensor_mask, dtype=bool),
        )
        sensor_sample = Sample(
            spectrum=sensor_spectrum,
            sensor_id=sensor_id,
            band_meta=band_meta,
            quality_masks={"sensor_mask": sensor_mask} if sensor_mask is not None else {},
            ancillary=noise_meta or {},
        )
        pairs.append(
            PairBatch(
                lab_spectrum=lab_spec,
                sensor_sample=sensor_sample,
                noise_cfg=noise_meta,
            )
        )
    return pairs


def build_emit_pairs(
    lab_batch: Sequence[_LabSpectrum],
    *,
    srf: str = "emit",
    srf_jitter_cfg: SRFJitterConfig | None = None,
    noise_cfg: NoiseConfig | None = None,
) -> list[PairBatch]:
    """Project high-resolution lab spectra onto EMIT bands and bundle pairs."""

    if len(lab_batch) == 0:
        return []

    lab_wl, lab_vals, _ = _normalize_lab_batch(lab_batch)
    try:
        sensor_srf = srfs.get_sensor_srf(srf)
    except FileNotFoundError:
        sensor_srf = None
    if sensor_srf is None:
        from alchemi.srf.emit import build_emit_sensor_srf
        from alchemi.srf.registry import register_sensor_srf

        sensor_srf = build_emit_sensor_srf(wavelength_grid_nm=lab_wl)  # type: ignore[arg-type]
        register_sensor_srf(sensor_srf)
    sensor_srf = _ensure_sensor_srf(sensor_srf)
    sensor_srf = _maybe_jitter_srf(sensor_srf, srf_jitter_cfg)
    sensor, centers = resample_values_with_srf(lab_vals, lab_wl, sensor_srf)
    if sensor.ndim == 1:
        sensor = sensor[None, :]

    cfg = noise_cfg or NoiseConfig()
    rng = cfg.generator()
    sensor_noisy = _apply_noise(sensor, cfg, rng=rng)

    sensor_wl = centers.astype(np.float64, copy=True)

    sensor_name = _sensor_identifier(sensor_srf)
    return _pairs_from_projection(
        lab_batch,
        lab_wl,
        sensor_noisy,
        sensor_wl,
        sensor_id=sensor_name,
        noise_levels=cfg.levels(sensor_wl.size),
        sensor_mask=getattr(sensor_srf, "valid_mask", None),
        band_widths_nm=getattr(sensor_srf, "band_widths_nm", None),
    )


def build_enmap_pairs(
    lab_batch: Sequence[_LabSpectrum],
    *,
    cache_dir: str | Path | None = "data/srf",
    noise_level_rel_vnir: float = 0.0,
    noise_level_rel_swir: float = 0.0,
    srf_jitter_cfg: SRFJitterConfig | None = None,
    noise_cfg: NoiseConfig | None = None,
) -> list[PairBatch]:
    """Project lab spectra to the EnMAP VNIR+SWIR grid with optional noise."""

    if len(lab_batch) == 0:
        return []

    lab_wl, lab_vals, _ = _normalize_lab_batch(lab_batch)
    cache_root = cache_dir if cache_dir is not None else "data/srf"
    try:
        sensor_srf = srfs.get_sensor_srf("enmap")
    except FileNotFoundError:
        sensor_srf = None
    if sensor_srf is None:
        from alchemi.srf.enmap import build_enmap_sensor_srf

        sensor_srf = build_enmap_sensor_srf(cache_dir=cache_root)
    sensor_srf = _ensure_sensor_srf(sensor_srf)
    sensor_srf = _maybe_jitter_srf(sensor_srf, srf_jitter_cfg)
    sensor, centers = resample_values_with_srf(lab_vals, lab_wl, sensor_srf)
    mask = getattr(sensor_srf, "valid_mask", None)

    override_levels = np.where(
        centers <= 999.0,
        float(noise_level_rel_vnir),
        float(noise_level_rel_swir),
    )
    cfg = noise_cfg or NoiseConfig()
    rng = cfg.generator()
    sensor_noisy = _apply_noise(sensor, cfg, rng=rng, override_levels=override_levels)

    sensor_name = _sensor_identifier(sensor_srf)
    return _pairs_from_projection(
        lab_batch,
        lab_wl,
        sensor_noisy,
        centers,
        sensor_id=sensor_name,
        sensor_mask=mask,
        noise_levels=override_levels,
        band_widths_nm=getattr(sensor_srf, "band_widths_nm", None),
    )


def build_avirisng_pairs(
    lab_batch: Sequence[_LabSpectrum],
    *,
    cache_dir: str | Path | None = None,
    noise: float | Sequence[float] | np.ndarray | None = None,
    srf_jitter_cfg: SRFJitterConfig | None = None,
    noise_cfg: NoiseConfig | None = None,
) -> list[PairBatch]:
    """Project lab spectra onto the AVIRIS-NG grid and optionally inject noise."""

    if len(lab_batch) == 0:
        return []

    lab_wl, lab_vals, _ = _normalize_lab_batch(lab_batch)
    try:
        sensor_srf = srfs.get_sensor_srf("aviris-ng")
    except FileNotFoundError:
        sensor_srf = None
    if sensor_srf is None:
        from alchemi.srf.avirisng import build_avirisng_sensor_srf

        sensor_srf = build_avirisng_sensor_srf(cache_dir=cache_dir)
    sensor_srf = _ensure_sensor_srf(sensor_srf)
    centers = np.asarray(sensor_srf.band_centers_nm, dtype=np.float64)
    mask = getattr(sensor_srf, "valid_mask", None)

    sensor_srf = _maybe_jitter_srf(sensor_srf, srf_jitter_cfg)
    sensor, _ = resample_values_with_srf(lab_vals, lab_wl, sensor_srf)
    cfg = noise_cfg or NoiseConfig()
    rng = cfg.generator()
    sensor_noisy = _apply_noise(sensor, cfg, rng=rng, override_levels=noise)

    sensor_name = _sensor_identifier(sensor_srf)
    return _pairs_from_projection(
        lab_batch,
        lab_wl,
        sensor_noisy,
        centers,
        sensor_id=sensor_name,
        sensor_mask=mask,
        noise_levels=cfg.levels(centers.size, override=noise),
        band_widths_nm=getattr(sensor_srf, "band_widths_nm", None),
    )


# Backwards compatibility alias retained for older tests/configs.
build_emits_pairs = build_emit_pairs


__all__ = [
    "NoiseConfig",
    "PairBatch",
    "SRFJitterConfig",
    "build_avirisng_pairs",
    "build_emit_pairs",
    "build_emits_pairs",
    "build_enmap_pairs",
]
