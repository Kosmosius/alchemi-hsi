"""Spectrum data model.

See the design doc's "Data and metadata model" section for the scientific
contract governing wavelength grids, spectral quantities, and unit handling.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Tuple

import numpy as np
from numpy.typing import NDArray

KindLiteral = Literal["radiance", "reflectance", "BT"]

WAVELENGTH_MONOTONICITY_EPS = 1e-9
MIN_WAVELENGTH_SEPARATION_EPS = 1e-12
REFLECTANCE_MIN = -0.05
REFLECTANCE_MAX = 2.0


def _normalize_kind(kind: KindLiteral | str) -> KindLiteral:
    normalized = str(kind).strip().lower()
    if normalized == "bt":
        normalized = "BT"
    if normalized not in {"radiance", "reflectance", "BT"}:
        raise ValueError(f"Unsupported spectrum kind: {kind!r}")
    return normalized  # type: ignore[return-value]


def _ensure_sorted(
    wavelength_nm: NDArray[np.floating], values: NDArray[np.floating]
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    order = np.argsort(wavelength_nm)
    sorted_wavelengths = wavelength_nm[order]
    sorted_values = values[order]
    return sorted_wavelengths, sorted_values


@dataclass
class Spectrum:
    """Canonical spectral sample with a strictly increasing wavelength grid."""

    wavelength_nm: NDArray[np.floating]
    values: NDArray[np.floating]
    kind: KindLiteral

    def __post_init__(self) -> None:
        self.wavelength_nm = np.asarray(self.wavelength_nm, dtype=float)
        self.values = np.asarray(self.values, dtype=float)
        self.kind = _normalize_kind(self.kind)
        self.validate()

    def validate(self) -> None:
        if self.wavelength_nm.ndim != 1:
            raise ValueError("wavelength_nm must be 1-D")
        if self.values.shape != self.wavelength_nm.shape:
            raise ValueError("values must match wavelength grid shape")

        diffs = np.diff(self.wavelength_nm)
        if np.any(diffs < -WAVELENGTH_MONOTONICITY_EPS):
            raise ValueError("wavelength_nm must increase monotonically")
        if np.any(diffs <= MIN_WAVELENGTH_SEPARATION_EPS):
            raise ValueError("wavelength_nm must be strictly increasing with separation")
        if np.any(~np.isfinite(self.wavelength_nm)):
            raise ValueError("wavelength_nm must be finite")

        if np.any(~np.isfinite(self.values)):
            raise ValueError("values must be finite")

        if self.kind == "reflectance":
            min_val = np.nanmin(self.values)
            max_val = np.nanmax(self.values)
            if min_val < REFLECTANCE_MIN or max_val > REFLECTANCE_MAX:
                raise ValueError(
                    "Reflectance values are expected to remain within ["
                    f"{REFLECTANCE_MIN}, {REFLECTANCE_MAX}]"
                )

    @classmethod
    def from_microns(
        cls,
        wavelength_um: NDArray[np.floating],
        values: NDArray[np.floating],
        *,
        kind: KindLiteral,
    ) -> "Spectrum":
        wavelength_nm = np.asarray(wavelength_um, dtype=float) * 1_000.0
        values_arr = np.asarray(values, dtype=float)
        if _normalize_kind(kind) == "radiance":
            values_arr = values_arr / 1_000.0
        wavelength_nm, values_arr = _ensure_sorted(wavelength_nm, values_arr)
        return cls(wavelength_nm=wavelength_nm, values=values_arr, kind=kind)

    @classmethod
    def from_wavenumber(
        cls,
        wavenumber_cm: NDArray[np.floating],
        values: NDArray[np.floating],
        *,
        kind: KindLiteral,
    ) -> "Spectrum":
        wavenumber_cm = np.asarray(wavenumber_cm, dtype=float)
        if np.any(wavenumber_cm <= 0):
            raise ValueError("wavenumber_cm must be positive")
        wavelength_nm = 1.0e7 / wavenumber_cm
        values_arr = np.asarray(values, dtype=float)
        if _normalize_kind(kind) == "radiance":
            values_arr = values_arr * 1.0e7 / np.square(wavelength_nm)
        wavelength_nm, values_arr = _ensure_sorted(wavelength_nm, values_arr)
        return cls(wavelength_nm=wavelength_nm, values=values_arr, kind=kind)

    def copy_with(self, *, kind: KindLiteral | None = None) -> "Spectrum":
        new_kind = _normalize_kind(kind) if kind is not None else self.kind
        return replace(self, kind=new_kind)
