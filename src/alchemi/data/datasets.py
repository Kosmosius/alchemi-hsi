from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ..types import Sample


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
