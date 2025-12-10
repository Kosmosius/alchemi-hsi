"""Quantity-kind tagging and unit normalisation utilities.

This module centralises lightweight type checks and conversions between the
quantity kinds defined in :mod:`alchemi.types`. Conversions here are intended to
be minimal (unit scaling only) so that physics routines and IO adapters can rely
on a single source of truth for acceptable units.
"""

from __future__ import annotations

from enum import Enum
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from alchemi.types import (
    QuantityKind,
    RadianceUnits,
    ReflectanceUnits,
    TemperatureUnits,
    ValueUnits,
)

# ---------------------------------------------------------------------------
# Quantity kind helpers
# ---------------------------------------------------------------------------


def is_radiance(kind: QuantityKind) -> bool:
    return kind == QuantityKind.RADIANCE


def is_reflectance(kind: QuantityKind) -> bool:
    return kind in {
        QuantityKind.REFLECTANCE,
        QuantityKind.TOA_REFLECTANCE,
        QuantityKind.SURFACE_REFLECTANCE,
    }


def is_brightness_temp(kind: QuantityKind) -> bool:
    return kind == QuantityKind.BRIGHTNESS_T


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

_UNIT_ALIASES: dict[ValueUnits, Iterable[str]] = {
    ValueUnits.RADIANCE_W_M2_SR_NM: (
        RadianceUnits.W_M2_SR_NM.value,
        "w/m^2/sr/nm",
    ),
    ValueUnits.RADIANCE_W_M2_SR_UM: (
        RadianceUnits.W_M2_SR_UM.value,
        "w/m^2/sr/um",
        "w/m2/sr/um",
    ),
    ValueUnits.REFLECTANCE_FRACTION: (
        ReflectanceUnits.FRACTION.value,
        "unitless",
        "dimensionless",
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

_EXPECTED_UNITS: dict[QuantityKind, tuple[ValueUnits, ...]] = {
    QuantityKind.RADIANCE: (
        ValueUnits.RADIANCE_W_M2_SR_NM,
        ValueUnits.RADIANCE_W_M2_SR_UM,
    ),
    QuantityKind.TOA_REFLECTANCE: (
        ValueUnits.REFLECTANCE_FRACTION,
        ValueUnits.REFLECTANCE_PERCENT,
    ),
    QuantityKind.SURFACE_REFLECTANCE: (
        ValueUnits.REFLECTANCE_FRACTION,
        ValueUnits.REFLECTANCE_PERCENT,
    ),
    QuantityKind.REFLECTANCE: (
        ValueUnits.REFLECTANCE_FRACTION,
        ValueUnits.REFLECTANCE_PERCENT,
    ),
    QuantityKind.BRIGHTNESS_T: (
        ValueUnits.TEMPERATURE_K,
        ValueUnits.TEMPERATURE_C,
        ValueUnits.TEMPERATURE_F,
    ),
}


def _unit_token(unit: str) -> str:
    token = unit.strip().lower().replace("·", "").replace(" ", "")
    token = token.replace("^", "").replace("⁻", "-")
    return token


def normalize_units(
    units: ValueUnits | RadianceUnits | ReflectanceUnits | TemperatureUnits | str,
    quantity: QuantityKind,
) -> ValueUnits:
    """Normalise arbitrary unit labels to canonical :class:`ValueUnits`.

    Parameters
    ----------
    units:
        Enum instance or string describing the input units. Strings are matched
        case-insensitively against a small alias table.
    quantity:
        Quantity kind the units should correspond to.
    """

    if isinstance(units, ValueUnits):
        normalized = units
    elif isinstance(units, Enum):
        normalized = ValueUnits(units.value)
    else:
        token = _unit_token(str(units))
        normalized = None
        for candidate, aliases in _UNIT_ALIASES.items():
            if token in {_unit_token(alias) for alias in aliases}:
                normalized = candidate
                break
        if normalized is None:
            msg = f"Unrecognised units: {units!r}"
            raise ValueError(msg)

    expected = _EXPECTED_UNITS.get(quantity)
    if expected and normalized not in expected:
        msg = f"Units {normalized.value!r} incompatible with quantity {quantity.value!r}"
        raise ValueError(msg)
    return normalized


def normalize_values_to_canonical(
    values: NDArray[np.float64],
    units: ValueUnits,
    quantity: QuantityKind,
) -> tuple[NDArray[np.float64], ValueUnits]:
    """Return ``values`` converted to the canonical units for ``quantity``."""

    if quantity == QuantityKind.RADIANCE:
        if units == ValueUnits.RADIANCE_W_M2_SR_UM:
            return (
                scale_radiance_between_wavelength_units(
                    values, units, ValueUnits.RADIANCE_W_M2_SR_NM
                ),
                ValueUnits.RADIANCE_W_M2_SR_NM,
            )
        return values, ValueUnits.RADIANCE_W_M2_SR_NM

    if quantity in {
        QuantityKind.REFLECTANCE,
        QuantityKind.TOA_REFLECTANCE,
        QuantityKind.SURFACE_REFLECTANCE,
    }:
        if units == ValueUnits.REFLECTANCE_PERCENT:
            return values / 100.0, ValueUnits.REFLECTANCE_FRACTION
        return values, ValueUnits.REFLECTANCE_FRACTION

    if quantity == QuantityKind.BRIGHTNESS_T:
        if units == ValueUnits.TEMPERATURE_C:
            return values + 273.15, ValueUnits.TEMPERATURE_K
        if units == ValueUnits.TEMPERATURE_F:
            return (values - 32.0) * (5.0 / 9.0) + 273.15, ValueUnits.TEMPERATURE_K
        return values, ValueUnits.TEMPERATURE_K

    return values, units


# ---------------------------------------------------------------------------
# Radiance scaling helpers
# ---------------------------------------------------------------------------


def scale_radiance_between_wavelength_units(
    values: NDArray[np.float64],
    from_unit: RadianceUnits | ValueUnits,
    to_unit: RadianceUnits | ValueUnits,
) -> NDArray[np.float64]:
    """Scale radiance densities between per-µm and per-nm conventions."""

    from_val = ValueUnits(getattr(from_unit, "value", from_unit))
    to_val = ValueUnits(getattr(to_unit, "value", to_unit))
    if from_val == to_val:
        return np.asarray(values, dtype=np.float64)

    if from_val == ValueUnits.RADIANCE_W_M2_SR_UM and to_val == ValueUnits.RADIANCE_W_M2_SR_NM:
        return np.asarray(values, dtype=np.float64) / 1000.0
    if from_val == ValueUnits.RADIANCE_W_M2_SR_NM and to_val == ValueUnits.RADIANCE_W_M2_SR_UM:
        return np.asarray(values, dtype=np.float64) * 1000.0

    msg = f"Unsupported radiance conversion: {from_val.value} -> {to_val.value}"
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Wavenumber <-> wavelength conversions with radiance rescaling
# ---------------------------------------------------------------------------


def radiance_wavenumber_cm1_to_wavelength_nm(
    radiance_per_cm1: NDArray[np.float64],
    wavenumber_cm1: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert radiance defined per wavenumber to per-nanometre units.

    Parameters
    ----------
    radiance_per_cm1:
        Radiance density expressed per ``cm⁻¹`` (e.g., W·m⁻²·sr⁻¹·(cm⁻¹)⁻¹).
    wavenumber_cm1:
        Wavenumber grid in ``cm⁻¹``. Must be strictly positive but need not be
        monotonic; the transformation is applied element-wise.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(wavelength_nm, radiance_per_nm)`` where the radiance is scaled by the
        Jacobian ``|dν̃/dλ_nm| = 1e7 / λ_nm²`` to preserve energy per bin.
    """

    k = np.asarray(wavenumber_cm1, dtype=np.float64)
    if np.any(k <= 0):
        raise ValueError("Wavenumbers must be positive")

    wavelength_nm = wavenumber_cm1_to_wavelength_nm(k)
    jacobian = 1.0e7 / np.square(wavelength_nm)
    radiance_nm = np.asarray(radiance_per_cm1, dtype=np.float64) * jacobian
    return wavelength_nm, radiance_nm


def radiance_wavelength_nm_to_wavenumber_cm1(
    radiance_per_nm: NDArray[np.float64],
    wavelength_nm: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert radiance defined per nanometre to per-wavenumber units."""

    wavelength_nm_arr = np.asarray(wavelength_nm, dtype=np.float64)
    if np.any(wavelength_nm_arr <= 0):
        raise ValueError("Wavelengths must be positive")

    wavenumber_cm1 = 1.0e7 / wavelength_nm_arr
    jacobian = np.square(wavelength_nm_arr) / 1.0e7
    radiance_cm1 = np.asarray(radiance_per_nm, dtype=np.float64) * jacobian
    return wavenumber_cm1, radiance_cm1


# ---------------------------------------------------------------------------
# Wavelength conversions
# ---------------------------------------------------------------------------


def wavelength_um_to_nm(wavelength_um: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(wavelength_um, dtype=np.float64) * 1000.0


def wavelength_nm_to_um(wavelength_nm: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(wavelength_nm, dtype=np.float64) / 1000.0


def wavenumber_cm1_to_wavelength_nm(k_cm1: NDArray[np.float64]) -> NDArray[np.float64]:
    k = np.asarray(k_cm1, dtype=np.float64)
    if np.any(k <= 0):
        raise ValueError("Wavenumbers must be positive")
    wavelength_m = 1.0 / (k * 100.0)
    return wavelength_m * 1e9
