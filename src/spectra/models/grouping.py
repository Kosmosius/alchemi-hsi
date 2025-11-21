from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

GroupingMode = Literal["contiguous", "kmeans"]


@dataclass
class GroupingConfig:
    G: int = 4
    mode: GroupingMode = "contiguous"


def _contiguous_groups(B: int, G: int) -> list[list[int]]:
    groups: list[list[int]] = []
    sizes = [B // G + (1 if i < B % G else 0) for i in range(G)]
    start = 0
    for size in sizes:
        end = start + size
        groups.append(list(range(start, end)))
        start = end
    return [g for g in groups if g]


def _kmeans_groups(
    B: int, G: int, wavelengths_nm: torch.Tensor | None, spectra_sample: torch.Tensor | None
) -> list[list[int]] | None:
    if spectra_sample is None or wavelengths_nm is None:
        return None
    if spectra_sample.ndim != 2:
        raise ValueError("spectra_sample must be (samples, bands)")

    cov = torch.cov(spectra_sample.T)
    if wavelengths_nm.numel() != B:
        return None
    centroids = torch.linspace(
        wavelengths_nm.min(), wavelengths_nm.max(), G, device=wavelengths_nm.device
    )
    centroids = centroids.unsqueeze(1)
    bands_pos = wavelengths_nm.unsqueeze(1)
    assignments = torch.zeros(B, dtype=torch.long, device=wavelengths_nm.device)
    for _ in range(5):
        dist = (bands_pos - centroids.T).pow(2)
        dist = dist + (1 - torch.diag(cov)).unsqueeze(1)
        assignments = dist.argmin(dim=1)
        counts = torch.bincount(assignments, minlength=G)
        if (counts == 0).any():
            return None
        centroids = torch.stack(
            [bands_pos[assignments == i].mean(dim=0) for i in range(G)]
        ).unsqueeze(1)
    groups: list[list[int]] = []
    for i in range(G):
        groups.append(assignments.eq(i).nonzero(as_tuple=True)[0].tolist())
    return groups


def make_groups(
    B: int,
    G: int,
    mode: GroupingMode = "contiguous",
    wavelengths_nm: torch.Tensor | None = None,
    spectra_sample: torch.Tensor | None = None,
) -> list[list[int]]:
    if G <= 0:
        raise ValueError("G must be positive")
    if mode == "contiguous":
        return _contiguous_groups(B, G)
    if mode == "kmeans":
        groups = _kmeans_groups(B, G, wavelengths_nm, spectra_sample)
        if groups is None:
            return _contiguous_groups(B, G)
        return groups
    raise ValueError(f"Unknown grouping mode: {mode}")


__all__ = ["GroupingConfig", "make_groups"]
