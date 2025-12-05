"""Spectral response function utilities.

Normalization and validation routines follow the design doc's guidance in the
"Data and metadata model" section so that flat spectra remain flat after SRF
convolution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .spectrum import Spectrum


@dataclass
class SRFMatrix:
    """Matrix-form SRFs sampled on a common wavelength grid."""

    wavelength_nm: NDArray[np.floating]
    matrix: NDArray[np.floating]

    def __post_init__(self) -> None:
        self.wavelength_nm = np.asarray(self.wavelength_nm, dtype=float)
        self.matrix = np.asarray(self.matrix, dtype=float)
        if self.wavelength_nm.ndim != 1:
            raise ValueError("wavelength_nm must be 1-D")
        if self.matrix.ndim != 2:
            raise ValueError("matrix must be 2-D (bands x wavelengths)")
        if self.matrix.shape[1] != self.wavelength_nm.shape[0]:
            raise ValueError("matrix column count must match wavelength grid length")

    def normalize_rows_trapz(self, *, area_tol: float = 1e-12) -> None:
        """Normalize each SRF row using trapezoidal integration."""

        integrals = np.trapz(self.matrix, x=self.wavelength_nm, axis=1)
        if np.any(~np.isfinite(integrals)):
            raise ValueError("SRF integrals must be finite")
        if np.any(integrals <= area_tol):
            raise ValueError("SRF rows must integrate to a positive area before normalization")
        self.matrix = self.matrix / integrals[:, np.newaxis]

    def assert_nonnegative(self, *, tol: float = 0.0) -> None:
        """Ensure there are no negative response values beyond tolerance."""

        min_val = float(np.min(self.matrix))
        if min_val < -tol:
            raise ValueError(f"SRF matrix contains negative entries below tolerance: {min_val}")

    def assert_flat_spectrum_preserved(self, spectrum: Spectrum, *, tol: float = 1e-6) -> None:
        """Verify a flat spectrum remains flat after SRF convolution."""

        if spectrum.band_count != self.wavelength_nm.shape[0]:
            raise ValueError("Spectrum length must match SRF wavelength grid")
        values = np.asarray(spectrum.values, dtype=float)
        if values.ndim != 1:
            values = values.reshape(-1, spectrum.band_count)[0]
        span = float(np.ptp(values))
        baseline = float(np.mean(values))
        if span > tol:
            raise ValueError("Spectrum must be approximately flat for this check")
        band_values = np.trapz(self.matrix * values, x=self.wavelength_nm, axis=1)
        allowed = tol * max(1.0, abs(baseline))
        if np.any(np.abs(band_values - baseline) > allowed):
            raise ValueError("Flat spectrum is not preserved by SRF matrix")
