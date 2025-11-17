"""Alignment-specific data augmentation transforms."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from alchemi.srf.synthetic import ProjectedSpectrum, SyntheticSensorConfig, project_lab_to_synthetic


class RandomSensorProject:
    """Project lab spectra onto randomized synthetic sensor responses."""

    def __init__(self, config: SyntheticSensorConfig) -> None:
        self.config = config
        self._rng = np.random.default_rng(config.seed)

    def __call__(
        self, lab_batch: Sequence[tuple[Sequence[float], Sequence[float]]]
    ) -> list[ProjectedSpectrum]:
        outputs: list[ProjectedSpectrum] = []
        for values, axis in lab_batch:
            seed = int(self._rng.integers(0, np.iinfo(np.int32).max))
            proj = project_lab_to_synthetic(values, axis, self.config, seed=seed)
            outputs.append(proj)
        return outputs
