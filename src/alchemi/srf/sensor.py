"""Canonical sensor SRF container and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


def _validate_wavelength_grid(wavelengths_nm: np.ndarray) -> NDArray[np.float64]:
    wl = np.asarray(wavelengths_nm, dtype=np.float64)
    if wl.ndim != 1:
        raise ValueError("wavelength_grid_nm must be 1-D")
    if wl.size < 2:
        raise ValueError("wavelength_grid_nm must contain at least two entries")
    diffs = np.diff(wl)
    if np.any(diffs <= 0):
        raise ValueError("wavelength_grid_nm must be strictly increasing")
    return wl


def _grid_deltas(wavelengths_nm: NDArray[np.float64]) -> NDArray[np.float64]:
    diffs = np.diff(wavelengths_nm)
    deltas = np.empty_like(wavelengths_nm)
    deltas[0] = diffs[0]
    deltas[-1] = diffs[-1]
    if wavelengths_nm.size > 2:
        deltas[1:-1] = 0.5 * (diffs[:-1] + diffs[1:])
    return deltas


def normalize_srf_rows(srfs: np.ndarray, wavelengths_nm: np.ndarray) -> NDArray[np.float64]:
    """Normalize SRF rows so they integrate to 1 under trapezoidal spacing."""

    wl = _validate_wavelength_grid(wavelengths_nm)
    arr = np.asarray(srfs, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != wl.shape[0]:
        raise ValueError("srfs must be a 2-D array matching the wavelength grid")
    if np.any(arr < 0):
        raise ValueError("SRF responses must be non-negative")

    deltas = _grid_deltas(wl)
    row_scale = arr @ deltas
    if np.any(~np.isfinite(row_scale)) or np.any(row_scale <= 0.0):
        raise ValueError("SRF rows must integrate to a positive finite area")

    normalized = arr / row_scale[:, None]
    return np.asarray(normalized, dtype=np.float64)


@dataclass
class SensorSRF:
    """SRF bundle on a common wavelength grid for a sensor."""

    wavelength_grid_nm: NDArray[np.float64]
    srfs: NDArray[np.float64]  # [bands, samples]
    band_centers_nm: NDArray[np.float64] | None = None
    band_widths_nm: NDArray[np.float64] | None = None
    band_ids: NDArray[np.float64] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        wl = _validate_wavelength_grid(self.wavelength_grid_nm)
        arr = np.asarray(self.srfs, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != wl.shape[0]:
            raise ValueError("srfs must be 2-D with columns matching wavelength grid")
        if arr.shape[0] == 0:
            raise ValueError("srfs must contain at least one band")
        if np.any(arr < 0):
            raise ValueError("SRF responses must be non-negative")

        self.wavelength_grid_nm = wl
        self.srfs = arr

        n_bands = arr.shape[0]
        if self.band_centers_nm is not None:
            centers = np.asarray(self.band_centers_nm, dtype=np.float64)
            if centers.shape != (n_bands,):
                raise ValueError("band_centers_nm must match the number of SRF rows")
            self.band_centers_nm = centers

        if self.band_widths_nm is not None:
            widths = np.asarray(self.band_widths_nm, dtype=np.float64)
            if widths.shape != (n_bands,):
                raise ValueError("band_widths_nm must match the number of SRF rows")
            self.band_widths_nm = widths

        if self.band_ids is not None:
            ids = np.asarray(self.band_ids)
            if ids.shape != (n_bands,):
                raise ValueError("band_ids must match the number of SRF rows")
            self.band_ids = ids

        if self.meta is None:
            self.meta = {}

    def normalized(self) -> "SensorSRF":
        """Return a copy with SRF rows normalized to unit area."""

        normalized_srfs = normalize_srf_rows(self.srfs, self.wavelength_grid_nm)
        return SensorSRF(
            wavelength_grid_nm=self.wavelength_grid_nm.copy(),
            srfs=normalized_srfs,
            band_centers_nm=None if self.band_centers_nm is None else self.band_centers_nm.copy(),
            band_widths_nm=None if self.band_widths_nm is None else self.band_widths_nm.copy(),
            band_ids=None if self.band_ids is None else self.band_ids.copy(),
            meta=dict(self.meta),
        )


__all__ = ["SensorSRF", "normalize_srf_rows"]
