from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from alchemi.utils.integrate import np_integrate as _np_integrate


class SpectrumKind(str, Enum):
    RADIANCE = "radiance"  # W·m^-2·sr^-1·nm^-1
    REFLECTANCE = "reflectance"  # [0,1]
    BT = "brightness_temp"  # Kelvin


@dataclass
class WavelengthGrid:
    nm: NDArray[np.float64]  # [B]

    def __post_init__(self) -> None:
        a = np.asarray(self.nm, dtype=np.float64)
        if a.ndim != 1 or np.any(np.diff(a) <= 0):
            raise ValueError("Wavelength grid must be strictly increasing 1-D array (nm)")
        self.nm = a


@dataclass
class Spectrum:
    wavelengths: WavelengthGrid
    values: NDArray[np.float64]  # [B]
    kind: SpectrumKind
    units: str
    mask: NDArray[np.bool_] | None = None  # [B] boolean
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        v = np.asarray(self.values, dtype=np.float64)
        if v.ndim != 1 or v.shape[0] != self.wavelengths.nm.shape[0]:
            raise ValueError("values length must match wavelengths")
        self.values = v
        if self.mask is not None:
            m = np.asarray(self.mask, dtype=bool)
            if m.shape != v.shape:
                raise ValueError("mask shape mismatch")
            self.mask = m

    def masked(self) -> Spectrum:
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
    centers_nm: NDArray[np.float64]  # [B]
    bands_nm: list[NDArray[np.float64]]  # len B
    bands_resp: list[NDArray[np.float64]]  # len B
    version: str = "v1"
    cache_key: str | None = None
    bad_band_mask: NDArray[np.bool_] | None = None
    bad_band_windows_nm: Sequence[tuple[float, float]] | None = None

    def row_integrals(self) -> NDArray[np.float64]:
        integrals = [
            float(
                _np_integrate(
                    np.asarray(resp, dtype=np.float64),
                    np.asarray(nm, dtype=np.float64),
                )
            )
            for nm, resp in zip(self.bands_nm, self.bands_resp, strict=True)
        ]
        return np.asarray(integrals, dtype=np.float64)

    def normalize_trapz(self) -> SRFMatrix:
        bands_nm: list[np.ndarray] = []
        bands_resp: list[np.ndarray] = []
        for nm, resp in zip(self.bands_nm, self.bands_resp, strict=True):
            nm_arr = np.asarray(nm, dtype=np.float64)
            resp_arr = np.asarray(resp, dtype=np.float64)
            area = float(_np_integrate(resp_arr, nm_arr))
            if not np.isfinite(area) or area <= 0.0:
                msg = "SRF bands must integrate to a positive finite area"
                raise ValueError(msg)
            bands_nm.append(nm_arr.copy())
            bands_resp.append(resp_arr / area)

        return SRFMatrix(
            self.sensor,
            np.asarray(self.centers_nm, dtype=np.float64).copy(),
            bands_nm,
            bands_resp,
            version=self.version,
            cache_key=self.cache_key,
            bad_band_mask=None if self.bad_band_mask is None else self.bad_band_mask.copy(),
            bad_band_windows_nm=self.bad_band_windows_nm,
        )


@dataclass
class SampleMeta:
    sensor_id: str
    row: int
    col: int
    datetime: Any | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"sensor": self.sensor_id, "row": self.row, "col": self.col}
        if self.datetime is not None:
            data["datetime"] = self.datetime
        data.update(self.extras)
        return data


@dataclass
class Sample:
    spectrum: Spectrum
    meta: SampleMeta | dict[str, Any]

    def __post_init__(self) -> None:
        if isinstance(self.meta, SampleMeta):
            self.meta = self.meta.as_dict()
        else:
            self.meta = dict(self.meta)
