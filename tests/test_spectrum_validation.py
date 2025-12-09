import logging

import numpy as np
import pytest

from alchemi.types import (
    BT_PLAUSIBLE_MAX_K,
    BT_PLAUSIBLE_MIN_K,
    QuantityKind,
    RadianceUnits,
    REFLECTANCE_MAX_EPS,
    ReflectanceUnits,
    Spectrum,
    TemperatureUnits,
    ValueUnits,
    WavelengthGrid,
)


@pytest.fixture
def wavelengths() -> WavelengthGrid:
    return WavelengthGrid(np.linspace(400, 700, 5))


def test_reflectance_valid(wavelengths: WavelengthGrid) -> None:
    values = np.linspace(0.0, 1.0, 5)
    Spectrum.from_reflectance(wavelengths, values, units=ReflectanceUnits.FRACTION)


def test_reflectance_out_of_bounds(wavelengths: WavelengthGrid) -> None:
    values = np.array([-0.1, 0.2, 0.3, 0.4, 0.5])
    with pytest.raises(ValueError, match="Reflectance values must be within"):
        Spectrum(
            wavelengths=wavelengths,
            values=values,
            kind=QuantityKind.REFLECTANCE,
            units=ReflectanceUnits.FRACTION,
        )

    values = np.array([0.0, 0.2, 1.0 + REFLECTANCE_MAX_EPS + 1e-4, 0.4, 0.5])
    with pytest.raises(ValueError, match="Reflectance values must be within"):
        Spectrum(
            wavelengths=wavelengths,
            values=values,
            kind=QuantityKind.REFLECTANCE,
            units=ReflectanceUnits.FRACTION,
        )


def test_radiance_non_negative(wavelengths: WavelengthGrid) -> None:
    values = np.array([0.0, 0.2, 0.3, -0.1, 0.5])
    with pytest.raises(ValueError, match="Radiance values must be non-negative"):
        Spectrum(
            wavelengths=wavelengths,
            values=values,
            kind=QuantityKind.RADIANCE,
            units=RadianceUnits.W_M2_SR_NM,
        )


def test_brightness_temperature_positive(wavelengths: WavelengthGrid) -> None:
    values = np.array([200.0, 250.0, 300.0, 350.0, 0.0])
    with pytest.raises(ValueError, match="> 0 K"):
        Spectrum.from_brightness_temperature(wavelengths, values, units=TemperatureUnits.KELVIN)


def test_brightness_temperature_warning_outside_plausible_range(
    wavelengths: WavelengthGrid, caplog: pytest.LogCaptureFixture
) -> None:
    values = np.array([BT_PLAUSIBLE_MIN_K - 5.0, 200.0, BT_PLAUSIBLE_MAX_K + 5.0, 220.0, 230.0])
    with caplog.at_level(logging.WARNING):
        Spectrum.from_brightness_temperature(wavelengths, values)
    assert any("plausible range" in record.message for record in caplog.records)


def test_invalid_quantity_units_pairing(wavelengths: WavelengthGrid) -> None:
    values = np.linspace(0.0, 1.0, 5)
    with pytest.raises(ValueError, match="incompatible"):
        Spectrum(
            wavelengths=wavelengths,
            values=values,
            kind=QuantityKind.REFLECTANCE,
            units=RadianceUnits.W_M2_SR_NM,
        )

    with pytest.raises(ValueError, match="incompatible"):
        Spectrum(
            wavelengths=wavelengths,
            values=values,
            kind=QuantityKind.RADIANCE,
            units=ValueUnits.REFLECTANCE_FRACTION,
        )


def test_values_shape_mismatch(wavelengths: WavelengthGrid) -> None:
    values = np.linspace(0.0, 1.0, 4)
    with pytest.raises(ValueError, match="wavelengths length"):
        Spectrum(
            wavelengths=wavelengths,
            values=values,
            kind=QuantityKind.RADIANCE,
            units=RadianceUnits.W_M2_SR_NM,
        )


def test_mask_shape_mismatch(wavelengths: WavelengthGrid) -> None:
    values = np.linspace(0.0, 1.0, 5)
    mask = np.array([True, False, True])
    with pytest.raises(ValueError, match="mask shape"):
        Spectrum(
            wavelengths=wavelengths,
            values=values,
            kind=QuantityKind.RADIANCE,
            units=RadianceUnits.W_M2_SR_NM,
            mask=mask,
        )
