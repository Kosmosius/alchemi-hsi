"""Data augmentation utilities for spectral datasets."""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Sequence

import numpy as np
import torch

from alchemi.align.transforms import RandomSensorProject
from alchemi.spectral import BandMetadata, Sample, Spectrum
from alchemi.srf.synthetic import SyntheticSensorConfig
from alchemi.types import QuantityKind
from alchemi.spectral.srf import SRFMatrix, SRFProvenance

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


class SyntheticSensorProject:
    """Project high-resolution lab spectra onto randomized synthetic sensors.

    This wrapper produces canonical :class:`~alchemi.spectral.sample.Sample`
    instances with :class:`~alchemi.spectral.sample.BandMetadata` and dense
    SRF matrices so downstream ingest/tokenizers can operate without custom
    alignment code.
    """

    def __init__(
        self,
        cfg: SyntheticSensorConfig,
        *,
        sensor_id: str = "synthetic",
        quantity_kind: QuantityKind = QuantityKind.REFLECTANCE,
    ) -> None:
        self.cfg = cfg
        self.sensor_id = sensor_id
        self.quantity_kind = quantity_kind
        self._project = RandomSensorProject(cfg)

    def __call__(self, spectrum: Spectrum) -> Sample:
        proj = self._project([(spectrum.values, spectrum.wavelength_nm)])[0]
        band_meta = BandMetadata(
            center_nm=proj.centers_nm,
            width_nm=proj.fwhm_nm,
            valid_mask=np.ones_like(proj.centers_nm, dtype=bool),
            srf_source=self.sensor_id,
            srf_provenance=SRFProvenance.SYNTHETIC.value,
            srf_approximate=False,
            width_from_default=False,
        )
        identity_srf = np.eye(proj.centers_nm.shape[0], dtype=np.float64)
        srf_matrix = SRFMatrix(wavelength_nm=proj.centers_nm, matrix=identity_srf)
        sample = Sample(
            spectrum=Spectrum(
                wavelength_nm=proj.centers_nm,
                values=proj.values.astype(np.float32, copy=False),
                kind=self.quantity_kind,
                mask=band_meta.valid_mask,
            ),
            sensor_id=self.sensor_id,
            band_meta=band_meta,
            srf_matrix=srf_matrix,
            ancillary={"srf_axis_nm": proj.srf_axis_nm},
        )
        return sample
