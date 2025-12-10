"""Spectral response function utilities.

Normalization and validation routines follow the design doc's guidance in the
"Data and metadata model" section so that flat spectra remain flat after SRF
convolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .spectrum import Spectrum
from alchemi.types import SRFMatrix as LegacySRFMatrix


@dataclass
class SRFMatrix:
    """Dense SRF matrix sampled on a shared wavelength grid.

    The matrix is shaped ``(bands, wavelengths)`` and paired with
    ``wavelength_nm`` so resampling and physics helpers can apply the Section-4
    invariants:

    * Rows represent band responses on the shared grid.
    * :meth:`normalize_rows_trapz` enforces per-band unit area using trapezoidal
      integration.
    * :meth:`assert_nonnegative` and :meth:`assert_flat_spectrum_preserved`
      provide quick checks for non-negative responses and conservation of a flat
      spectrum after convolution.
    """

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


class SRFProvenance(str, Enum):
    """Provenance flag for SRF construction."""

    OFFICIAL = "official"
    GAUSSIAN = "gaussian"
    NONE = "none"


@dataclass
class SensorSRF:
    """Canonical SRF payload returned by the sensor SRF registry.

    Each ``SensorSRF`` bundles per-band SRFs sampled on a common wavelength grid
    for a specific ``sensor_id``. Band centres (and optional widths) capture the
    mission's nominal wavelengths, while ``valid_mask`` and ``meta`` (e.g.,
    ``bad_band_windows_nm``) carry QA hints. ``as_matrix`` exposes a dense
    :class:`SRFMatrix` suitable for physics utilities and resampling.
    """

    sensor_id: str
    wavelength_grid_nm: NDArray[np.floating]
    srfs: NDArray[np.floating]
    band_centers_nm: NDArray[np.floating]
    band_widths_nm: NDArray[np.floating] | None = None
    provenance: SRFProvenance = SRFProvenance.OFFICIAL
    valid_mask: NDArray[np.bool_] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.sensor_id = str(self.sensor_id).lower()
        self.wavelength_grid_nm = np.asarray(self.wavelength_grid_nm, dtype=float)
        self.srfs = np.asarray(self.srfs, dtype=float)
        self.band_centers_nm = np.asarray(self.band_centers_nm, dtype=float)
        if self.band_widths_nm is not None:
            self.band_widths_nm = np.asarray(self.band_widths_nm, dtype=float)
        if self.valid_mask is not None:
            self.valid_mask = np.asarray(self.valid_mask, dtype=bool)

        if self.wavelength_grid_nm.ndim != 1:
            raise ValueError("wavelength_grid_nm must be 1-D")
        if self.srfs.ndim != 2:
            raise ValueError("srfs must be 2-D (bands x wavelengths)")
        if self.srfs.shape[1] != self.wavelength_grid_nm.shape[0]:
            raise ValueError("srfs column count must match wavelength grid length")
        band_count = self.srfs.shape[0]
        if self.band_centers_nm.shape[0] != band_count:
            raise ValueError("band_centers_nm length must match SRF band count")
        if self.band_widths_nm is not None and self.band_widths_nm.shape[0] != band_count:
            raise ValueError("band_widths_nm length must match SRF band count")
        if self.valid_mask is not None and self.valid_mask.shape[0] != band_count:
            raise ValueError("valid_mask length must match SRF band count")

    @property
    def band_count(self) -> int:
        return int(self.srfs.shape[0])

    def as_matrix(self) -> SRFMatrix:
        """Return a dense :class:`SRFMatrix` view of the SRFs."""

        return SRFMatrix(wavelength_nm=self.wavelength_grid_nm, matrix=self.srfs)

    def to_matrix(self) -> LegacySRFMatrix:
        """Return a legacy :class:`alchemi.types.SRFMatrix` representation."""

        bands_nm = [np.asarray(self.wavelength_grid_nm, dtype=float).copy() for _ in range(self.band_count)]
        bands_resp = [np.asarray(row, dtype=float).copy() for row in self.srfs]
        return LegacySRFMatrix(
            sensor=self.sensor_id,
            centers_nm=np.asarray(self.band_centers_nm, dtype=float).copy(),
            bands_nm=bands_nm,
            bands_resp=bands_resp,
            version=str(self.meta.get("version", "v1")) if self.meta else "v1",
            cache_key=self.meta.get("cache_key") if self.meta else None,
            bad_band_mask=np.asarray(self.valid_mask, dtype=bool).copy() if self.valid_mask is not None else None,
            bad_band_windows_nm=self.meta.get("bad_band_windows_nm") if self.meta else None,
        )
