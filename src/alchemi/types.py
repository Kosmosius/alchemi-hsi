from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from alchemi.utils.integrate import np_integrate as _np_integrate


logger = logging.getLogger(__name__)


class SpectrumKind(str, Enum):
    RADIANCE = "radiance"  # W·m^-2·sr^-1·nm^-1
    REFLECTANCE = "reflectance"  # [0,1]
    BT = "brightness_temp"  # Kelvin


REFLECTANCE_MAX_EPS = 1e-3
BT_PLAUSIBLE_MIN_K = 150.0
BT_PLAUSIBLE_MAX_K = 400.0
EXPECTED_UNITS: dict[SpectrumKind, set[str]] = {
    SpectrumKind.RADIANCE: {
        "w·m^-2·sr^-1·nm^-1",
        "w m-2 sr-1 nm-1",
        "w/m^2/sr/nm",
    },
    SpectrumKind.REFLECTANCE: {"", "unitless", "dimensionless", "reflectance", "1"},
    SpectrumKind.BT: {"k", "kelvin"},
}


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

        self._validate_values()
        self._warn_on_units()

    def _validate_values(self) -> None:
        if self.kind == SpectrumKind.REFLECTANCE:
            if np.any(self.values < 0) or np.any(self.values > 1.0 + REFLECTANCE_MAX_EPS):
                msg = "Reflectance values must be within [0, 1 + eps]"
                raise ValueError(msg)
        elif self.kind == SpectrumKind.BT:
            if np.any(self.values <= 0):
                msg = "Brightness temperature values must be > 0 K"
                raise ValueError(msg)
            finite_values = self.values[np.isfinite(self.values)]
            if finite_values.size > 0:
                min_val = float(np.nanmin(finite_values))
                max_val = float(np.nanmax(finite_values))
                if min_val < BT_PLAUSIBLE_MIN_K or max_val > BT_PLAUSIBLE_MAX_K:
                    logger.warning(
                        "Brightness temperature values fall outside plausible range [%s, %s] K",
                        BT_PLAUSIBLE_MIN_K,
                        BT_PLAUSIBLE_MAX_K,
                    )
        elif self.kind == SpectrumKind.RADIANCE:
            if np.any(self.values < 0):
                msg = "Radiance values must be non-negative"
                raise ValueError(msg)

    def _warn_on_units(self) -> None:
        expected = EXPECTED_UNITS.get(self.kind)
        if expected is None:
            return
        normalized_units = self.units.strip().lower()
        if normalized_units not in expected:
            logger.warning("Unexpected units '%s' for spectrum kind '%s'", self.units, self.kind.value)

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

    def __post_init__(self) -> None:
        num_bands = len(self.bands_nm)
        if len(self.bands_resp) != num_bands:
            msg = "bands_resp length must match bands_nm"
            raise ValueError(msg)
        if len(self.centers_nm) != num_bands:
            msg = "centers_nm length must match number of bands"
            raise ValueError(msg)

        centers_nm = np.asarray(self.centers_nm, dtype=np.float64)
        bands_nm: list[np.ndarray] = []
        bands_resp: list[np.ndarray] = []

        for idx, (nm, resp) in enumerate(zip(self.bands_nm, self.bands_resp, strict=True)):
            nm_arr = np.asarray(nm, dtype=np.float64)
            resp_arr = np.asarray(resp, dtype=np.float64)

            if nm_arr.ndim != 1:
                msg = f"bands_nm[{idx}] must be 1-D"
                raise ValueError(msg)
            if resp_arr.ndim != 1:
                msg = f"bands_resp[{idx}] must be 1-D"
                raise ValueError(msg)
            if nm_arr.shape[0] != resp_arr.shape[0]:
                msg = f"bands_nm[{idx}] and bands_resp[{idx}] must have the same length"
                raise ValueError(msg)

            bands_nm.append(nm_arr)
            bands_resp.append(resp_arr)

        self.centers_nm = centers_nm
        self.bands_nm = bands_nm
        self.bands_resp = bands_resp

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
