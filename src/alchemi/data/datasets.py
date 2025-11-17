from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ..types import Sample


class SpectrumDataset(Dataset):
    def __init__(self, samples: list[Sample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i: int) -> dict[str, Any]:
        s = self.samples[i]
        w = torch.from_numpy(s.spectrum.wavelengths.nm.astype("float32"))
        v = torch.from_numpy(s.spectrum.values.astype("float32"))
        m = torch.from_numpy(
            (
                s.spectrum.mask if s.spectrum.mask is not None else np.ones_like(s.spectrum.values)
            ).astype("bool")
        )
        return {
            "wavelengths": w,
            "values": v,
            "mask": m,
            "kind": s.spectrum.kind.value,
            "meta": s.meta,
        }


class PairingDataset(Dataset):
    def __init__(self, field: list[Sample], lab_conv: list[Sample]):
        assert len(field) == len(lab_conv)
        self.field, self.lab = field, lab_conv

    def __len__(self):
        return len(self.field)

    def __getitem__(self, i):
        def pack(s: Sample) -> dict[str, Any]:
            w = torch.from_numpy(s.spectrum.wavelengths.nm.astype("float32"))
            v = torch.from_numpy(s.spectrum.values.astype("float32"))
            m = torch.from_numpy(
                (
                    s.spectrum.mask
                    if s.spectrum.mask is not None
                    else np.ones_like(s.spectrum.values)
                ).astype("bool")
            )
            return {
                "wavelengths": w,
                "values": v,
                "mask": m,
                "kind": s.spectrum.kind.value,
                "meta": s.meta,
            }

        return {"field": pack(self.field[i]), "lab": pack(self.lab[i])}
