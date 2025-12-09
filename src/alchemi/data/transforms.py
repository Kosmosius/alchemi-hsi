"""Data augmentation utilities for spectral datasets."""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Sequence

import numpy as np
import torch

from alchemi.spectral import Spectrum

TensorLike = torch.Tensor


class SpectralNoise:
    """Additive Gaussian noise applied to spectra."""

    def __init__(self, sigma: float = 0.002):
        self.sigma = sigma

    def __call__(self, x: TensorLike) -> TensorLike:
        return x + self.sigma * torch.randn_like(x)


class RandomBandDropout:
    """Randomly mask a subset of spectral bands."""

    def __init__(self, p: float = 0.05):
        self.p = p

    def __call__(self, mask: TensorLike) -> TensorLike:
        drop = (torch.rand_like(mask.float()) < self.p).bool()
        return mask & ~drop


class GeometricAugment:
    """Optional spatial flips/rotations for chips."""

    def __init__(self, enable: bool = True):
        self.enable = enable

    def __call__(self, chip: torch.Tensor) -> torch.Tensor:
        if not self.enable:
            return chip
        if random.random() < 0.5:
            chip = torch.flip(chip, dims=[0])
        if random.random() < 0.5:
            chip = torch.flip(chip, dims=[1])
        k = random.randint(0, 3)
        if k:
            chip = torch.rot90(chip, k, dims=[0, 1])
        return chip


class SRFJitter:
    """Apply small random perturbations to emulate sensor SRF variation."""

    def __init__(self, centers_nm: Sequence[float], jitter_nm: float = 1.0):
        self.centers_nm = np.asarray(centers_nm, dtype=np.float64)
        self.jitter_nm = jitter_nm
        self.rng = np.random.default_rng()

    def __call__(self, spectrum: Spectrum) -> Spectrum:
        offsets = self.rng.uniform(-self.jitter_nm, self.jitter_nm, size=self.centers_nm.shape)
        perturbed = self.centers_nm + offsets
        values = np.interp(perturbed, spectrum.wavelength_nm, spectrum.values)
        # TODO: swap to physics.resampling once spectral/type unification lands.
        return Spectrum(wavelength_nm=perturbed, values=values, kind=spectrum.kind)


def compose(
    transforms: Iterable[Callable[[torch.Tensor], torch.Tensor]],
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Compose a list of callables into a single transform."""

    def _inner(x: torch.Tensor) -> torch.Tensor:
        for fn in transforms:
            x = fn(x)
        return x

    return _inner
