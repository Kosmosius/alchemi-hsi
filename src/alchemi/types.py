# TODO: migrated from legacy structure – reconcile with new design.
from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray

from alchemi.utils.integrate import np_integrate as _np_integrate

logger = logging.getLogger(__name__)


class QuantityKind(str, Enum):
    RADIANCE = "radiance"
    REFLECTANCE = "reflectance"
    BRIGHTNESS_T = "brightness_temperature"
    BT = "brightness_temperature"  # Alias for backwards compatibility


class RadianceUnits(str, Enum):
    W_M2_SR_NM = "W·m⁻²·sr⁻¹·nm⁻¹"
    W_M2_SR_UM = "W·m⁻²·sr⁻¹·µm⁻¹"


class ReflectanceUnits(str, Enum):
    FRACTION = "fraction"
    PERCENT = "percent"


class TemperatureUnits(str, Enum):
    KELVIN = "K"
    CELSIUS = "C"
    FAHRENHEIT = "F"


class ValueUnits(str, Enum):
    """Canonical value units across supported quantity kinds."""

    RADIANCE_W_M2_SR_NM = RadianceUnits.W_M2_SR_NM.value
    RADIANCE_W_M2_SR_UM = RadianceUnits.W_M2_SR_UM.value
    REFLECTANCE_FRACTION = ReflectanceUnits.FRACTION.value
    REFLECTANCE_PERCENT = ReflectanceUnits.PERCENT.value
    TEMPERATURE_K = TemperatureUnits.KELVIN.value
    TEMPERATURE_C = TemperatureUnits.CELSIUS.value
    TEMPERATURE_F = TemperatureUnits.FAHRENHEIT.value


# Backwards-compatible alias; prefer QuantityKind going forward.
SpectrumKind = QuantityKind


# Wavelength grid validation tolerances
WAVELENGTH_GRID_MONOTONICITY_EPS = 0.0
"""Absolute tolerance (in nm) for wavelength monotonicity checks."""

WAVELENGTH_GRID_DUPLICATE_EPS = 1e-12
"""Absolute tolerance (in nm) for detecting repeated wavelengths."""

REFLECTANCE_MAX_EPS = 1e-3
BT_PLAUSIBLE_MIN_K = 150.0
BT_PLAUSIBLE_MAX_K = 400.0

_QUANTITY_ALIASES = {
    "radiance": QuantityKind.RADIANCE,
    "reflectance": QuantityKind.REFLECTANCE,
    "brightness_temperature": QuantityKind.BRIGHTNESS_T,
    "brightness_temp": QuantityKind.BRIGHTNESS_T,
    "bt": QuantityKind.BRIGHTNESS_T,
}

_UNIT_ALIASES: dict[ValueUnits, Iterable[str]] = {
    ValueUnits.RADIANCE_W_M2_SR_NM: (
        RadianceUnits.W_M2_SR_NM.value,
        "W·m^-2·sr^-1·nm^-1",
        "w m-2 sr-1 nm-1",
        "w/m^2/sr/nm",
        "w·m⁻²·sr⁻¹·nm⁻¹",
    ),
    ValueUnits.REFLECTANCE_FRACTION: (
        ReflectanceUnits.FRACTION.value,
        "unitless",
        "dimensionless",
        "reflectance",
        "",
        "1",
    ),
    ValueUnits.REFLECTANCE_PERCENT: (
        ReflectanceUnits.PERCENT.value,
        "%",
        "percent",
    ),
    ValueUnits.TEMPERATURE_K: (
        TemperatureUnits.KELVIN.value,
        "kelvin",
        "k",
    ),
    ValueUnits.TEMPERATURE_C: (
        TemperatureUnits.CELSIUS.value,
        "celsius",
        "c",
    ),
    ValueUnits.TEMPERATURE_F: (
        TemperatureUnits.FAHRENHEIT.value,
        "fahrenheit",
        "f",
    ),
}

_EXPECTED_UNITS: dict[QuantityKind, set[ValueUnits]] = {
    QuantityKind.RADIANCE: {ValueUnits.RADIANCE_W_M2_SR_NM, ValueUnits.RADIANCE_W_M2_SR_UM},
    QuantityKind.REFLECTANCE: {ValueUnits.REFLECTANCE_FRACTION, ValueUnits.REFLECTANCE_PERCENT},
    QuantityKind.BRIGHTNESS_T: {
        ValueUnits.TEMPERATURE_K,
        ValueUnits.TEMPERATURE_C,
        ValueUnits.TEMPERATURE_F,
    },
}


def _wavelength_unit_scale(units: str | None) -> float:
    """Return the multiplicative scale to convert the given unit to nm."""

    if units is None:
        return 1.0
    token = units.strip().lower().replace(" ", "")
    if token in {"nm", "nanometer", "nanometre"}:
        return 1.0
    if token in {"um", "µm", "micrometer", "micrometre"}:
        return 1e3
    if token in {"ang", "angstrom", "å", "a"}:
        return 0.1
    msg = f"Unsupported wavelength unit: {units!r}"
    raise ValueError(msg)


def _normalize_quantity_kind(kind: QuantityKind | SpectrumKind | str) -> QuantityKind:
    if isinstance(kind, QuantityKind):
        return kind
    if isinstance(kind, Enum):
        return _normalize_quantity_kind(kind.value)

    key = str(kind).strip().lower()
    normalized = key.replace(" ", "_")
    if normalized in _QUANTITY_ALIASES:
        if not isinstance(kind, QuantityKind):
            warnings.warn(
                "Quantity kind strings are deprecated; use QuantityKind enum values instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        return _QUANTITY_ALIASES[normalized]
    msg = f"Unsupported quantity kind: {kind!r}"
    raise ValueError(msg)


def _unit_token(unit: str) -> str:
    token = unit.strip().lower()
    token = token.replace("·", "").replace(" ", "")
    token = token.replace("⁻", "-")
    token = token.replace("^", "")
    return token


def _normalize_value_units(
    units: ValueUnits | RadianceUnits | ReflectanceUnits | TemperatureUnits | str,
    kind: QuantityKind,
) -> ValueUnits:
    if isinstance(units, Enum):
        try:
            normalized_units = ValueUnits(getattr(units, "value", units))
        except ValueError as exc:
            msg = f"Unrecognised units: {units!r}"
            raise ValueError(msg) from exc
    else:
        if units is None:
            msg = "Units must be provided"
            raise ValueError(msg)
        unit_str = str(units)
        matched = None
        token = _unit_token(unit_str)
        for value_unit, aliases in _UNIT_ALIASES.items():
            alias_tokens = {_unit_token(alias) for alias in aliases}
            if token in alias_tokens:
                matched = value_unit
                break
        if matched is None:
            msg = f"Unrecognised units: {unit_str!r}"
            raise ValueError(msg)
        warnings.warn(
            "Unit strings are deprecated; use ValueUnits / RadianceUnits / ReflectanceUnits / TemperatureUnits enums instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        normalized_units = matched

    expected = _EXPECTED_UNITS.get(kind)
    if expected is None:
        return normalized_units
    if normalized_units not in expected:
        msg = f"Units {normalized_units.value!r} are incompatible with quantity {kind.value!r}"
        raise ValueError(msg)
    return normalized_units


@dataclass
class WavelengthGrid:
    nm: NDArray[np.float64]  # [B]

    def __post_init__(self) -> None:
        """
        Canonical wavelength grid stored in **nanometres**, strictly increasing.

        The grid is normalised to ``float64`` and validated to be 1-D with
        strictly increasing values (``λ[i+1] > λ[i]``). This class is the
        single source of truth for spectral axes across M1.
        """

        a = np.asarray(self.nm, dtype=np.float64)
        if a.ndim != 1:
            raise ValueError("Wavelength grid must be a 1-D array (nm)")
        if a.size == 0:
            raise ValueError("Wavelength grid must contain at least one entry")

        if a.size > 1:
            diffs = np.diff(a)
            if np.any(diffs <= WAVELENGTH_GRID_MONOTONICITY_EPS):
                raise ValueError("Wavelength grid must be strictly increasing (nm)")
            if np.any(np.isclose(diffs, 0.0, atol=WAVELENGTH_GRID_DUPLICATE_EPS)):
                raise ValueError("Wavelength grid must not contain repeated bands")

        self.nm = a

    @classmethod
    def from_any(cls, values: Sequence[float] | NDArray[np.floating], units: str | None = None) -> WavelengthGrid:
        """Create a grid from values expressed in nm, µm, or Ångström.

        Parameters
        ----------
        values:
            1-D wavelength coordinates.
        units:
            Unit label for ``values``. Supported forms are ``"nm"`` (default),
            ``"um"``/``"µm"``, and ``"angstrom"``/``"Å"``. Unknown units raise
            ``ValueError`` to avoid silent guesses.
        """

        scale = _wavelength_unit_scale(units)
        nm_values = np.asarray(values, dtype=np.float64) * scale
        return cls(nm_values)

    def to(self, unit: str) -> NDArray[np.float64]:
        """Return the wavelength grid converted to the requested unit."""

        scale = 1.0 / _wavelength_unit_scale(unit)
        return self.nm * scale


@dataclass
class Spectrum:
    """Canonical spectral sample coupling wavelengths, values, and metadata.

    ``Spectrum`` pairs a :class:`WavelengthGrid` (always stored in nanometres)
    with a 1-D value vector, an explicit :class:`QuantityKind`, and
    normalised units. Construction enforces shape alignment and optional mask
    compatibility so that downstream physics utilities can rely on consistent
    invariants across M1.
    """

    wavelengths: WavelengthGrid | Sequence[float] | NDArray[np.floating]
    values: NDArray[np.float64]  # [B]
    kind: QuantityKind | SpectrumKind | str
    units: ValueUnits | RadianceUnits | ReflectanceUnits | TemperatureUnits | str
    mask: NDArray[np.bool_] | None = None  # [B] boolean
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.wavelengths, WavelengthGrid):
            self.wavelengths = WavelengthGrid.from_any(self.wavelengths)

        v = np.asarray(self.values, dtype=np.float64)
        if v.ndim != 1:
            raise ValueError("values must be a 1-D array")
        if v.shape[0] != self.wavelengths.nm.shape[0]:
            raise ValueError("values length must match wavelengths")
        self.values = v

        if self.mask is not None:
            m = np.asarray(self.mask, dtype=bool)
            if m.shape != v.shape:
                raise ValueError("mask shape must match values")
            self.mask = m

        self.kind = _normalize_quantity_kind(self.kind)
        from alchemi.physics import units as qty_units

        normalized_units = qty_units.normalize_units(self.units, self.kind)
        values, canonical_units = qty_units.normalize_values_to_canonical(v, normalized_units, self.kind)
        self.values = np.asarray(values, dtype=np.float64)
        self.units = canonical_units
        self.meta = dict(self.meta)

        self._validate_values()

    def _validate_values(self) -> None:
        if self.kind == QuantityKind.REFLECTANCE:
            if np.any(self.values < 0) or np.any(self.values > 1.0 + REFLECTANCE_MAX_EPS):
                msg = "Reflectance values must be within [0, 1 + eps]"
                raise ValueError(msg)
        elif self.kind == QuantityKind.BRIGHTNESS_T:
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
        elif self.kind == QuantityKind.RADIANCE:
            if np.any(self.values < 0):
                msg = "Radiance values must be non-negative"
                raise ValueError(msg)

    @property
    def wavelength_nm(self) -> NDArray[np.float64]:
        """Return the underlying wavelength grid in nanometres."""

        return self.wavelengths.nm

    @property
    def band_count(self) -> int:
        """Number of spectral bands."""

        return int(self.values.shape[0])

    def copy_with(
        self,
        *,
        wavelengths: WavelengthGrid | Sequence[float] | NDArray[np.floating] | None = None,
        values: NDArray[np.float64] | None = None,
        mask: NDArray[np.bool_] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Spectrum:
        """Return a shallow copy with selected fields replaced."""

        return Spectrum(
            wavelengths=wavelengths if wavelengths is not None else self.wavelengths,
            values=values if values is not None else self.values,
            kind=self.kind,
            units=self.units,
            mask=mask if mask is not None else self.mask,
            meta=meta if meta is not None else dict(self.meta),
        )

    @classmethod
    def from_radiance(
        cls,
        wavelengths: WavelengthGrid,
        values: NDArray[np.float64],
        *,
        units: RadianceUnits | ValueUnits | str = RadianceUnits.W_M2_SR_NM,
        mask: NDArray[np.bool_] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Spectrum:
        return cls(
            wavelengths=wavelengths,
            values=values,
            kind=QuantityKind.RADIANCE,
            units=units,
            mask=mask,
            meta=meta or {},
        )

    @classmethod
    def from_reflectance(
        cls,
        wavelengths: WavelengthGrid,
        values: NDArray[np.float64],
        *,
        units: ReflectanceUnits | ValueUnits | str = ReflectanceUnits.FRACTION,
        mask: NDArray[np.bool_] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Spectrum:
        return cls(
            wavelengths=wavelengths,
            values=values,
            kind=QuantityKind.REFLECTANCE,
            units=units,
            mask=mask,
            meta=meta or {},
        )

    @classmethod
    def from_brightness_temperature(
        cls,
        wavelengths: WavelengthGrid,
        values: NDArray[np.float64],
        *,
        units: TemperatureUnits | ValueUnits | str = TemperatureUnits.KELVIN,
        mask: NDArray[np.bool_] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Spectrum:
        return cls(
            wavelengths=wavelengths,
            values=values,
            kind=QuantityKind.BRIGHTNESS_T,
            units=units,
            mask=mask,
            meta=meta or {},
        )

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


# TODO: Legacy SRFMatrix retained for compatibility; prefer alchemi.spectral.SRFMatrix.
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

    def normalize_rows_trapz(self) -> SRFMatrix:
        """Alias for :meth:`normalize_trapz` for backward compatibility.

        The registry helpers expect SRF matrices to be normalized per band.
        This convenience wrapper preserves the original interface used in the
        ALCHEMI design notes while delegating to ``normalize_trapz``.
        """

        return self.normalize_trapz()


# TODO: Legacy SampleMeta retained for compatibility; prefer alchemi.spectral.Sample.
@dataclass
class SampleMeta:
    """Metadata describing a per-pixel measurement.

    ``Sample`` instances created from a :class:`~alchemi.data.cube.Cube` should
    carry the originating sensor identifier (or SRF id), the source row/column
    indices, and any additional cube attributes that are meaningful at the
    pixel level.
    """

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


# TODO: Legacy Sample retained for compatibility; prefer alchemi.spectral.Sample.
@dataclass
class Sample:
    """A single measured spectrum (e.g. a pixel extracted from a cube).

    ``Sample`` is intentionally lightweight to accommodate lab-style spectra
    that may not originate from an image. When a ``Sample`` is derived from a
    :class:`~alchemi.data.cube.Cube` the spectrum's wavelength grid and
    :class:`SpectrumKind` **must** match the cube's spectral axis and
    value_kind. Spatial provenance (``row``/``col``) and sensor identifiers are
    captured in :class:`SampleMeta`.
    """

    spectrum: Spectrum
    meta: SampleMeta | dict[str, Any]

    def __post_init__(self) -> None:
        if isinstance(self.meta, SampleMeta):
            self.meta = self.meta.as_dict()
        else:
            self.meta = dict(self.meta)
