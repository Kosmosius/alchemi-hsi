from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from alchemi.tokens.registry import AxisUnit


@dataclass(slots=True)
class SyntheticAlignmentDataset:
    """Lightweight in-memory dataset producing paired lab and overhead spectra."""

    lab_wavelengths_nm: np.ndarray
    sensor_wavelengths_nm: np.ndarray
    lab_values: np.ndarray
    sensor_values: np.ndarray
    axis_unit: AxisUnit = "nm"

    def __len__(self) -> int:
        return int(self.lab_values.shape[0])

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        return self.lab_values[index], self.sensor_values[index]

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

        lab_values_arr = np.stack(lab_rows, axis=0)
        sensor_values_arr = np.stack(sensor_rows, axis=0)
        sensor_wavelengths = lab_wavelengths.copy()

        return cls(
            lab_wavelengths_nm=lab_wavelengths,
            sensor_wavelengths_nm=sensor_wavelengths,
            lab_values=lab_values_arr,
            sensor_values=sensor_values_arr,
            axis_unit=axis_unit,
        )

    def batch(self, indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        lab = np.asarray(self.lab_values[list(indices)], dtype=np.float64)
        sensor = np.asarray(self.sensor_values[list(indices)], dtype=np.float64)
        return lab, sensor
