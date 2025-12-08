from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from alchemi.spectral import Sample, Spectrum
from alchemi.tokens.registry import AxisUnit
from alchemi.types import QuantityKind, ReflectanceUnits


@dataclass(slots=True)
class SyntheticAlignmentDataset:
    """Lightweight in-memory dataset producing paired lab and overhead spectra."""

    lab_spectra: list[Spectrum]
    sensor_samples: list[Sample]
    axis_unit: AxisUnit = "nm"

    def __len__(self) -> int:
        return len(self.lab_spectra)

    def __getitem__(self, index: int) -> tuple[Spectrum, Sample]:
        return self.lab_spectra[index], self.sensor_samples[index]

    @classmethod
    def create(
        cls,
        *,
        num_pairs: int = 32,
        num_samples: int = 64,
        seed: int = 0,
        axis_unit: AxisUnit = "nm",
    ) -> "SyntheticAlignmentDataset":
        rng = np.random.default_rng(seed)
        lab_wavelengths = np.linspace(380.0, 2500.0, num_samples, dtype=np.float64)

        # Shared smooth baseline that every sample perturbs slightly.
        baseline = 0.6 + 0.4 * np.sin(np.linspace(0.0, 3.0 * np.pi, num_samples))
        lab_rows: list[np.ndarray] = []
        sensor_rows: list[np.ndarray] = []
        for _ in range(num_pairs):
            drift = rng.normal(loc=0.0, scale=0.02)
            slope = rng.normal(loc=1.0, scale=0.01)
            lab_noise = rng.normal(loc=0.0, scale=0.01, size=num_samples)
            sensor_noise = rng.normal(loc=0.0, scale=0.01, size=num_samples)

            lab_values = baseline * (1.0 + drift) + lab_noise
            sensor_values = slope * lab_values + sensor_noise

            lab_rows.append(lab_values.astype(np.float64, copy=False))
            sensor_rows.append(sensor_values.astype(np.float64, copy=False))

        lab_spectra: list[Spectrum] = []
        sensor_samples: list[Sample] = []
        for lab_values, sensor_values in zip(lab_rows, sensor_rows, strict=True):
            lab_spectrum = Spectrum(
                wavelength_nm=lab_wavelengths,
                values=lab_values,
                kind=QuantityKind.REFLECTANCE,
                units=ReflectanceUnits.FRACTION,
            )
            sensor_spectrum = Spectrum(
                wavelength_nm=lab_wavelengths,
                values=sensor_values,
                kind=QuantityKind.REFLECTANCE,
                units=ReflectanceUnits.FRACTION,
            )
            lab_spectra.append(lab_spectrum)
            sensor_samples.append(Sample(spectrum=sensor_spectrum, sensor_id="synthetic"))

        return cls(lab_spectra=lab_spectra, sensor_samples=sensor_samples, axis_unit=axis_unit)

    @property
    def lab_wavelengths_nm(self) -> np.ndarray:
        return np.asarray(self.lab_spectra[0].wavelength_nm, dtype=np.float64)

    @property
    def sensor_wavelengths_nm(self) -> np.ndarray:
        return np.asarray(self.sensor_samples[0].spectrum.wavelength_nm, dtype=np.float64)

    def batch(self, indices: Sequence[int]) -> tuple[list[Spectrum], list[Sample]]:
        lab = [self.lab_spectra[idx] for idx in indices]
        sensor = [self.sensor_samples[idx] for idx in indices]
        return lab, sensor
