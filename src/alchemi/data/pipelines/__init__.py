"""Pipeline builders combining datasets, tiling, and transforms."""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np

from torch.utils.data import DataLoader

from alchemi.config import DataConfig

from ..catalog import SceneCatalog
from ..datasets import (
    AvirisGasDataset,
    EmitGasDataset,
    EmitSolidsDataset,
    HytesDataset,
    SpectralEarthDataset,
    SyntheticSensorDataset,
)
from ..transforms import GeometricAugment, RandomBandDropout, SpectralNoise, compose
from ...srf.synthetic import SyntheticSensorConfig
from ...spectral import Sample, Spectrum
from ...types import QuantityKind


def _build_transforms(cfg: DataConfig) -> Callable:
    transforms: list = []
    if cfg.augmentation.noise_injection:
        transforms.append(SpectralNoise())
    if cfg.augmentation.band_masking:
        transforms.append(RandomBandDropout())
    return compose(transforms) if transforms else lambda x: x


def build_emit_solids_pipeline(
    data_cfg: DataConfig, *, catalog: SceneCatalog | None = None
) -> DataLoader:
    dataset = EmitSolidsDataset(
        split=data_cfg.splits.get("train", ["train"])[0],
        catalog=catalog or SceneCatalog(),
        transform=_build_transforms(data_cfg),
        patch_size=data_cfg.sample.patch_size,
        stride=data_cfg.sample.stride,
    )
    return DataLoader(dataset, batch_size=data_cfg.sample.batch_size, shuffle=True)


def build_emit_gas_pipeline(
    data_cfg: DataConfig, *, catalog: SceneCatalog | None = None
) -> DataLoader:
    dataset = EmitGasDataset(
        split=data_cfg.splits.get("train", ["train"])[0],
        catalog=catalog or SceneCatalog(),
        transform=_build_transforms(data_cfg),
    )
    return DataLoader(dataset, batch_size=data_cfg.sample.batch_size, shuffle=True)


def build_enmap_mae_pipeline(
    data_cfg: DataConfig, *, catalog: SceneCatalog | None = None
) -> DataLoader:
    dataset = SpectralEarthDataset(
        split=data_cfg.splits.get("train", ["train"])[0],
        catalog=catalog or SceneCatalog(),
        transform=_build_transforms(data_cfg),
        patch_size=data_cfg.sample.patch_size,
        stride=data_cfg.sample.stride,
    )
    augment = GeometricAugment(enable=data_cfg.augmentation.geometric_transforms)

    def _collate(batch: Iterable[dict]):
        return [augment(item) if hasattr(item, "shape") else item for item in batch]

    return DataLoader(
        dataset, batch_size=data_cfg.sample.batch_size, shuffle=True, collate_fn=_collate
    )


def _synthetic_lab_spectra(data_cfg: DataConfig) -> list[Sample]:
    cfg = data_cfg.synthetic_sensor
    axis = np.arange(cfg.wavelength_min, cfg.wavelength_max + cfg.resolution, cfg.resolution)
    rng = np.random.default_rng(cfg.seed)
    base_values = [np.sin(axis / 150.0), np.cos(axis / 210.0), rng.normal(scale=0.05, size=axis.size)]
    samples: list[Sample] = []
    for idx, values in enumerate(base_values):
        spectrum = Spectrum(wavelength_nm=axis, values=values.astype(np.float64), kind="reflectance")
        samples.append(Sample(spectrum=spectrum, sensor_id=f"lab_{idx}"))
    return samples


def build_synthetic_sensor_pipeline(
    data_cfg: DataConfig,
) -> DataLoader:
    if not data_cfg.synthetic_sensor.enabled:
        msg = "Synthetic sensor pipeline requested but synthetic_sensor.enabled is False"
        raise ValueError(msg)

    synth_cfg = data_cfg.synthetic_sensor
    highres_samples = _synthetic_lab_spectra(data_cfg)
    sensor_cfg = SyntheticSensorConfig(
        highres_axis_nm=highres_samples[0].spectrum.wavelength_nm,
        n_bands=synth_cfg.n_bands,
        center_jitter_nm=synth_cfg.center_jitter_nm,
        fwhm_range_nm=synth_cfg.fwhm_range_nm,
        shape=synth_cfg.shape,
        seed=synth_cfg.seed,
    )
    dataset = SyntheticSensorDataset(
        highres_samples,
        sensor_cfg,
        sensor_id=synth_cfg.sensor_id,
        quantity_kind=QuantityKind.REFLECTANCE,
    )

    return DataLoader(dataset, batch_size=data_cfg.sample.batch_size, shuffle=True)


def build_aviris_gas_pipeline(
    data_cfg: DataConfig, *, catalog: SceneCatalog | None = None
) -> DataLoader:
    dataset = AvirisGasDataset(
        split=data_cfg.splits.get("train", ["train"])[0],
        catalog=catalog or SceneCatalog(),
        transform=_build_transforms(data_cfg),
    )
    return DataLoader(dataset, batch_size=data_cfg.sample.batch_size, shuffle=True)


def build_hytes_pipeline(
    data_cfg: DataConfig, *, catalog: SceneCatalog | None = None
) -> DataLoader:
    dataset = HytesDataset(
        split=data_cfg.splits.get("train", ["train"])[0],
        catalog=catalog or SceneCatalog(),
        transform=_build_transforms(data_cfg),
        patch_size=data_cfg.sample.patch_size,
        stride=data_cfg.sample.stride,
    )
    return DataLoader(dataset, batch_size=data_cfg.sample.batch_size, shuffle=True)


__all__ = [
    "build_emit_solids_pipeline",
    "build_emit_gas_pipeline",
    "build_enmap_mae_pipeline",
    "build_aviris_gas_pipeline",
    "build_hytes_pipeline",
    "build_synthetic_sensor_pipeline",
]
