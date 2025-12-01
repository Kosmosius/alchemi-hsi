"""Pipeline builders combining datasets, tiling, and transforms."""

from __future__ import annotations

from typing import Callable, Iterable

from torch.utils.data import DataLoader

from alchemi.config import DataConfig

from ..catalog import SceneCatalog
from ..datasets import AvirisGasDataset, EmitGasDataset, EmitSolidsDataset, HytesDataset, SpectralEarthDataset
from ..transforms import GeometricAugment, RandomBandDropout, SpectralNoise, compose


def _build_transforms(cfg: DataConfig) -> Callable:
    transforms: list = []
    if cfg.augmentation.noise_injection:
        transforms.append(SpectralNoise())
    if cfg.augmentation.band_masking:
        transforms.append(RandomBandDropout())
    return compose(transforms) if transforms else lambda x: x


def build_emit_solids_pipeline(data_cfg: DataConfig, *, catalog: SceneCatalog | None = None) -> DataLoader:
    dataset = EmitSolidsDataset(
        split=data_cfg.splits.get("train", ["train"])[0],
        catalog=catalog or SceneCatalog(),
        transform=_build_transforms(data_cfg),
        patch_size=data_cfg.sample.patch_size,
        stride=data_cfg.sample.stride,
    )
    return DataLoader(dataset, batch_size=data_cfg.sample.batch_size, shuffle=True)


def build_emit_gas_pipeline(data_cfg: DataConfig, *, catalog: SceneCatalog | None = None) -> DataLoader:
    dataset = EmitGasDataset(
        split=data_cfg.splits.get("train", ["train"])[0],
        catalog=catalog or SceneCatalog(),
        transform=_build_transforms(data_cfg),
    )
    return DataLoader(dataset, batch_size=data_cfg.sample.batch_size, shuffle=True)


def build_enmap_mae_pipeline(data_cfg: DataConfig, *, catalog: SceneCatalog | None = None) -> DataLoader:
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

    return DataLoader(dataset, batch_size=data_cfg.sample.batch_size, shuffle=True, collate_fn=_collate)


def build_aviris_gas_pipeline(data_cfg: DataConfig, *, catalog: SceneCatalog | None = None) -> DataLoader:
    dataset = AvirisGasDataset(
        split=data_cfg.splits.get("train", ["train"])[0],
        catalog=catalog or SceneCatalog(),
        transform=_build_transforms(data_cfg),
    )
    return DataLoader(dataset, batch_size=data_cfg.sample.batch_size, shuffle=True)


def build_hytes_pipeline(data_cfg: DataConfig, *, catalog: SceneCatalog | None = None) -> DataLoader:
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
]
