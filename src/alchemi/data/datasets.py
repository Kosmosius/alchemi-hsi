from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ..types import Sample
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
