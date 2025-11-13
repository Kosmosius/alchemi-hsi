from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any

import numpy as np


class SpectrumKind(str, Enum):
    RADIANCE = "radiance"  # W·m^-2·sr^-1·nm^-1
    REFLECTANCE = "reflectance"  # [0,1]
    BT = "brightness_temp"  # Kelvin


@dataclass
class WavelengthGrid:
    nm: np.ndarray  # [B]

    def __post_init__(self):
        a = np.asarray(self.nm)
        if a.ndim != 1 or np.any(np.diff(a) <= 0):
            raise ValueError("Wavelength grid must be strictly increasing 1-D array (nm)")
        self.nm = a.astype(np.float64)


@dataclass
class Spectrum:
    wavelengths: WavelengthGrid
    values: np.ndarray  # [B]
    kind: SpectrumKind
    units: str
    mask: Optional[np.ndarray] = None  # [B] boolean
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        v = np.asarray(self.values)
        if v.ndim != 1 or v.shape[0] != self.wavelengths.nm.shape[0]:
            raise ValueError("values length must match wavelengths")
        self.values = v.astype(np.float64)
        if self.mask is not None:
            m = np.asarray(self.mask).astype(bool)
            if m.shape != v.shape:
                raise ValueError("mask shape mismatch")
            self.mask = m

    def masked(self) -> "Spectrum":
        if self.mask is None:
            return self
        keep = self.mask
        return Spectrum(
            WavelengthGrid(self.wavelengths.nm[keep]),
            self.values[keep],
            self.kind,
            self.units,
            None,
            self.meta.copy(),
        )


@dataclass
class SRFMatrix:
    sensor: str
    centers_nm: np.ndarray  # [B]
    bands_nm: list[np.ndarray]  # len B
    bands_resp: list[np.ndarray]  # len B
    version: str = "v1"
    cache_key: Optional[str] = None

    def row_integrals(self) -> np.ndarray:
        from numpy import trapezoid

        return np.array(
            [trapezoid(r, nm) for nm, r in zip(self.bands_nm, self.bands_resp)], dtype=np.float64
        )

    def normalize_trapz(self) -> "SRFMatrix":
        from numpy import trapezoid

        nr = []
        for nm, r in zip(self.bands_nm, self.bands_resp):
            area = float(trapezoid(r, nm))
            if area <= 0:
                raise ValueError("SRF area must be >0")
            nr.append(r / area)
        return SRFMatrix(
            self.sensor, self.centers_nm, self.bands_nm, nr, self.version, self.cache_key
        )


@dataclass
class SampleMeta:
    sensor_id: str
    row: Optional[int] = None
    col: Optional[int] = None
    datetime: Optional[str] = None
    georef: Optional[Dict[str, Any]] = None


@dataclass
class Sample:
    spectrum: Spectrum
    meta: SampleMeta
