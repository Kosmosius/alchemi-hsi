from __future__ import annotations

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable, Iterable, List

import numpy as np
import torch
from torch.utils.data import Dataset

from alchemi.spectral import Sample as SpectralSample
from alchemi.spectral import Spectrum

from ..types import Sample
from .adapters import iter_aviris_ng_pixels, iter_emit_pixels, iter_enmap_pixels, iter_hytes_pixels
from .catalog import SceneCatalog
from .tiling import iter_tiles
from .io import load_emit_l1b


class SpectrumDataset(Dataset[dict[str, Any]]):
    """Simple Dataset wrapper for lists of spectral samples."""

    def __init__(self, samples: Sequence[Sample]):
        self.samples = list(samples)

    def __len__(self) -> int:  # pragma: no cover - trivial container
        return len(self.samples)

    def __getitem__(self, i: int) -> dict[str, Any]:
        s = self.samples[i]
        w = torch.from_numpy(s.spectrum.wavelengths.nm.astype("float32"))
        v = torch.from_numpy(s.spectrum.values.astype("float32"))
        mask_source = (
            s.spectrum.mask
            if s.spectrum.mask is not None
            else np.ones_like(s.spectrum.values, dtype=bool)
        )
        m = torch.from_numpy(mask_source.astype("bool"))
        return {
            "wavelengths": w,
            "values": v,
            "mask": m,
            "kind": s.spectrum.kind.value,
            "meta": s.meta,
        }


class PairingDataset(Dataset[dict[str, dict[str, Any]]]):
    """Dataset yielding dictionaries containing paired field/lab samples."""

    def __init__(self, field: Sequence[Sample], lab_conv: Sequence[Sample]):
        if len(field) != len(lab_conv):
            msg = "field and lab_conv must share length"
            raise ValueError(msg)
        self.field = list(field)
        self.lab = list(lab_conv)

    def __len__(self) -> int:  # pragma: no cover - trivial container
        return len(self.field)

    def __getitem__(self, i: int) -> dict[str, dict[str, Any]]:
        def pack(s: Sample) -> dict[str, Any]:
            w = torch.from_numpy(s.spectrum.wavelengths.nm.astype("float32"))
            v = torch.from_numpy(s.spectrum.values.astype("float32"))
            mask_source = (
                s.spectrum.mask
                if s.spectrum.mask is not None
                else np.ones_like(s.spectrum.values, dtype=bool)
            )
            m = torch.from_numpy(mask_source.astype("bool"))
            return {
                "wavelengths": w,
                "values": v,
                "mask": m,
                "kind": s.spectrum.kind.value,
                "meta": s.meta,
            }

        return {"field": pack(self.field[i]), "lab": pack(self.lab[i])}


class RealMAEDataset(Dataset[dict[str, torch.Tensor]]):
    """Tiny dataset that samples patches from real cube fixtures for MAE runs."""

    _LOADERS: dict[str, Any] = {
        "emit_fixture": load_emit_l1b,
    }

    def __init__(
        self,
        dataset_name: str,
        path: str,
        *,
        patch_size: int = 2,
        max_patches: int = 32,
    ) -> None:
        if dataset_name not in self._LOADERS:
            msg = f"Unsupported dataset_name {dataset_name!r}"
            raise ValueError(msg)

        loader = self._LOADERS[dataset_name]
        ds = loader(path)

        if "radiance" not in ds:
            raise KeyError("Expected 'radiance' variable in dataset")

        data = ds["radiance"].values
        if data.ndim != 3:
            msg = "radiance data must be three-dimensional (y, x, band)"
            raise ValueError(msg)

        self.data = data.astype("float32", copy=False)
        self.wavelengths = torch.from_numpy(ds["wavelength_nm"].values.astype("float32"))
        band_mask = ds["band_mask"].values if "band_mask" in ds else None
        if band_mask is None:
            band_mask = torch.ones(self.data.shape[-1], dtype=torch.bool)
        else:
            band_mask = torch.from_numpy(band_mask.astype("bool"))
        self.band_mask = band_mask

        self.patch_size = max(1, patch_size)
        h, w, _ = self.data.shape
        effective_patch = min(self.patch_size, h, w)
        ys = range(0, h - effective_patch + 1, effective_patch)
        xs = range(0, w - effective_patch + 1, effective_patch)
        coords = [(y, x) for y in ys for x in xs]
        if not coords:
            coords = [(0, 0)]
        self.coords = coords[:max_patches]
        self.effective_patch = effective_patch

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        y, x = self.coords[index]
        patch = self.data[y : y + self.effective_patch, x : x + self.effective_patch]
        tokens = torch.from_numpy(patch.reshape(-1, patch.shape[-1]).astype("float32", copy=False))
        return {
            "tokens": tokens,
            "wavelengths": self.wavelengths,
            "band_mask": self.band_mask,
        }


# -----------------------------------------------------------------------------
# Catalog-backed datasets


def _pack_spectral_sample(sample: SpectralSample, transform: Callable[[torch.Tensor], torch.Tensor] | None) -> dict[str, Any]:
    values = torch.from_numpy(sample.spectrum.values.astype("float32"))
    if transform is not None:
        values = transform(values)
    return {
        "values": values,
        "wavelengths": torch.from_numpy(sample.spectrum.wavelength_nm.astype("float32")),
        "kind": sample.spectrum.kind,
        "quality_masks": {k: torch.from_numpy(v.astype("bool")) for k, v in sample.quality_masks.items()},
        "meta": {"sensor_id": sample.sensor_id, **sample.ancillary},
    }


class _BaseCatalogDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        *,
        split: str,
        sensor_id: str,
        task: str,
        catalog: SceneCatalog | None,
        transform: Callable[[torch.Tensor], torch.Tensor] | None,
    ) -> None:
        self.catalog = catalog or SceneCatalog()
        self.scene_paths = self.catalog.get_scenes(split, sensor_id, task)
        self.transform = transform
        self.samples: list[dict[str, Any]] = []

    def __len__(self) -> int:  # pragma: no cover - container
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]


class SpectralEarthDataset(_BaseCatalogDataset):
    """MAE pretraining dataset built from EnMAP-derived SpectralEarth patches."""

    def __init__(
        self,
        *,
        split: str,
        catalog: SceneCatalog | None = None,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        patch_size: int = 32,
        stride: int = 16,
    ) -> None:
        super().__init__(split=split, sensor_id="enmap", task="spectral_earth", catalog=catalog, transform=transform)
        for scene in self.scene_paths:
            for sample in iter_enmap_pixels(str(scene)):
                # Build chips using a synthetic 2x2 window to keep stubs lightweight.
                chip = sample.spectrum.values.reshape(1, 1, -1)
                for tile in iter_tiles(chip, patch_size, stride):
                    tensor_chip = torch.from_numpy(tile.data.astype("float32"))
                    if transform is not None:
                        tensor_chip = transform(tensor_chip)
                    self.samples.append({"chip": tensor_chip, "wavelengths": torch.from_numpy(sample.spectrum.wavelength_nm.astype("float32"))})


class EmitSolidsDataset(_BaseCatalogDataset):
    """EMIT solids dataset with paired L2A/L2B labels."""

    def __init__(
        self,
        *,
        split: str,
        catalog: SceneCatalog | None = None,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        patch_size: int = 32,
        stride: int = 16,
    ) -> None:
        super().__init__(split=split, sensor_id="emit", task="emit_solids", catalog=catalog, transform=transform)
        for scene in self.scene_paths:
            for sample in iter_emit_pixels(str(scene)):
                chip = sample.spectrum.values.reshape(1, 1, -1)
                for tile in iter_tiles(chip, patch_size, stride):
                    tensor_chip = torch.from_numpy(tile.data.astype("float32"))
                    if transform is not None:
                        tensor_chip = transform(tensor_chip)
                    self.samples.append({
                        "chip": tensor_chip,
                        "wavelengths": torch.from_numpy(sample.spectrum.wavelength_nm.astype("float32")),
                        "labels": sample.ancillary.get("labels", {}),
                    })


class EmitGasDataset(_BaseCatalogDataset):
    """Gas detection dataset built from EMIT teacher outputs (CTMF/DOAS)."""

    def __init__(
        self,
        *,
        split: str,
        catalog: SceneCatalog | None = None,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        super().__init__(split=split, sensor_id="emit", task="emit_gas", catalog=catalog, transform=transform)
        for scene in self.scene_paths:
            for sample in iter_emit_pixels(str(scene)):
                packed = _pack_spectral_sample(sample, transform)
                self.samples.append(packed)


class AvirisGasDataset(_BaseCatalogDataset):
    """Gas detection dataset for AVIRIS-NG scenes."""

    def __init__(
        self,
        *,
        split: str,
        catalog: SceneCatalog | None = None,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        super().__init__(split=split, sensor_id="aviris-ng", task="aviris_gas", catalog=catalog, transform=transform)
        for scene in self.scene_paths:
            for sample in iter_aviris_ng_pixels(str(scene)):
                self.samples.append(_pack_spectral_sample(sample, transform))


class HytesDataset(_BaseCatalogDataset):
    """LWIR brightness temperature chips."""

    def __init__(
        self,
        *,
        split: str,
        catalog: SceneCatalog | None = None,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        patch_size: int = 16,
        stride: int = 16,
    ) -> None:
        super().__init__(split=split, sensor_id="hytes", task="hytes_bt", catalog=catalog, transform=transform)
        for scene in self.scene_paths:
            for sample in iter_hytes_pixels(str(scene)):
                chip = sample.spectrum.values.reshape(1, 1, -1)
                for tile in iter_tiles(chip, patch_size, stride):
                    tensor_chip = torch.from_numpy(tile.data.astype("float32"))
                    if transform is not None:
                        tensor_chip = transform(tensor_chip)
                    self.samples.append({
                        "chip": tensor_chip,
                        "wavelengths": torch.from_numpy(sample.spectrum.wavelength_nm.astype("float32")),
                        "meta": sample.ancillary,
                    })
