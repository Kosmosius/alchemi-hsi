"""Alignment-specific data augmentation transforms."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from alchemi.srf import project_to_sensor
from alchemi.srf.synthetic import estimate_fwhm, rand_srf_grid
from alchemi.types import SRFMatrix


class RandomSensorProject:
    """Project spectra onto randomized synthetic sensor responses."""

    def __init__(
        self,
        grid_nm: Sequence[float],
        *,
        n_bands: int,
        center_jitter_nm: float,
        fwhm_range_nm: Iterable[float] | float,
        shape: str = "gaussian",
        seed: int | np.random.Generator | None = None,
    ) -> None:
        self.grid_nm = self._validate_grid(grid_nm)
        if n_bands <= 0:
            raise ValueError("n_bands must be positive")
        self.n_bands = int(n_bands)
        if center_jitter_nm < 0:
            raise ValueError("center_jitter_nm must be non-negative")
        self.center_jitter_nm = float(center_jitter_nm)
        self.fwhm_range_nm = fwhm_range_nm
        if shape not in {"gaussian", "box", "hamming"}:
            raise ValueError("shape must be one of {'gaussian','box','hamming'}")
        self.shape = shape
        if isinstance(seed, np.random.Generator):
            self._rng = seed
        else:
            self._rng = np.random.default_rng(seed)

    @staticmethod
    def _validate_grid(grid_nm: Sequence[float]) -> np.ndarray:
        wl = np.asarray(grid_nm, dtype=np.float64)
        if wl.ndim != 1:
            raise ValueError("grid_nm must be a 1-D sequence")
        if wl.size < 2 or np.any(np.diff(wl) <= 0):
            raise ValueError("grid_nm must be strictly increasing")
        return wl

    def _draw_matrix(self) -> SRFMatrix:
        _, matrix = rand_srf_grid(
            self.grid_nm,
            n_bands=self.n_bands,
            center_jitter_nm=self.center_jitter_nm,
            fwhm_range_nm=self.fwhm_range_nm,
            shape=self.shape,  # type: ignore[arg-type]
            seed=self._rng,
        )
        return matrix

    def __call__(self, spectrum: Sequence[float] | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = np.asarray(spectrum, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError("Spectrum must be a 1-D array")
        if values.shape[0] != self.grid_nm.shape[0]:
            raise ValueError("Spectrum length must match grid length")

        matrix = self._draw_matrix()
        projected = np.asarray(
            project_to_sensor(self.grid_nm, values, matrix.centers_nm, srf=matrix),
            dtype=np.float64,
        )
        centers = np.asarray(matrix.centers_nm, dtype=np.float64)
        fwhm_est = np.asarray(
            [estimate_fwhm(wl, resp) for wl, resp in zip(matrix.bands_nm, matrix.bands_resp, strict=True)],
            dtype=np.float64,
        )
        return projected, centers, fwhm_est
