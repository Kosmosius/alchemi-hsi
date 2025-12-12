"""Alignment-specific data augmentation transforms."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from alchemi.srf.synthetic import (
    ProjectedSpectrum,
    SyntheticSensorConfig,
    project_lab_to_synthetic,
    sample_synthetic_sensor,
)


class RandomSensorProject:
    """Project lab spectra onto randomized synthetic sensor responses."""

    def __init__(
        self, config: SyntheticSensorConfig, *, mode: str = "per_sample"
    ) -> None:
        self.config = config
        self.mode = mode
        if self.mode not in {"per_sample", "per_batch", "fixed"}:
            raise ValueError("mode must be one of 'per_sample', 'per_batch', or 'fixed'")

        self._rng = np.random.default_rng(config.seed)
        self._fixed_sensor = (
            sample_synthetic_sensor(config, rng=self._rng) if self.mode == "fixed" else None
        )

    def __call__(
        self, lab_batch: Sequence[tuple[Sequence[float], Sequence[float]]]
    ) -> list[ProjectedSpectrum]:
        outputs: list[ProjectedSpectrum] = []
        batch_sensor = None
        if self.mode == "per_batch":
            batch_sensor = sample_synthetic_sensor(self.config, rng=self._rng)

        for values, axis in lab_batch:
            if self.mode == "per_sample":
                seed = int(self._rng.integers(0, np.iinfo(np.int32).max))
                proj = project_lab_to_synthetic(values, axis, self.config, seed=seed)
            elif self.mode == "per_batch":
                assert batch_sensor is not None
                proj = project_lab_to_synthetic(values, axis, self.config, sensor=batch_sensor)
            else:  # fixed
                if self._fixed_sensor is None:
                    self._fixed_sensor = sample_synthetic_sensor(self.config, rng=self._rng)
                proj = project_lab_to_synthetic(values, axis, self.config, sensor=self._fixed_sensor)
            outputs.append(proj)
        return outputs
