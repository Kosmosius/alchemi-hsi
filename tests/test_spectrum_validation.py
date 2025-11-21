import logging

import numpy as np
import pytest

from alchemi.types import (
    BT_PLAUSIBLE_MAX_K,
    BT_PLAUSIBLE_MIN_K,
    REFLECTANCE_MAX_EPS,
    Spectrum,
    SpectrumKind,
    WavelengthGrid,
)


@pytest.fixture
def wavelengths() -> WavelengthGrid:
    return WavelengthGrid(np.linspace(400, 700, 5))


def test_reflectance_valid(wavelengths: WavelengthGrid) -> None:
    values = np.linspace(0.0, 1.0, 5)
    Spectrum(wavelengths, values, SpectrumKind.REFLECTANCE, "dimensionless")


def test_reflectance_out_of_bounds(wavelengths: WavelengthGrid) -> None:
    values = np.array([-0.1, 0.2, 0.3, 0.4, 0.5])
    with pytest.raises(ValueError, match="Reflectance values must be within"):
        Spectrum(wavelengths, values, SpectrumKind.REFLECTANCE, "dimensionless")

    values = np.array([0.0, 0.2, 1.0 + REFLECTANCE_MAX_EPS + 1e-4, 0.4, 0.5])
    with pytest.raises(ValueError, match="Reflectance values must be within"):
        Spectrum(wavelengths, values, SpectrumKind.REFLECTANCE, "dimensionless")


def test_radiance_non_negative(wavelengths: WavelengthGrid) -> None:
    values = np.array([0.0, 0.2, 0.3, -0.1, 0.5])
    with pytest.raises(ValueError, match="Radiance values must be non-negative"):
        Spectrum(wavelengths, values, SpectrumKind.RADIANCE, "W·m^-2·sr^-1·nm^-1")


def test_brightness_temperature_positive(wavelengths: WavelengthGrid) -> None:
    values = np.array([200.0, 250.0, 300.0, 350.0, 0.0])
    with pytest.raises(ValueError, match="> 0 K"):
        Spectrum(wavelengths, values, SpectrumKind.BT, "K")


def test_brightness_temperature_warning_outside_plausible_range(
    wavelengths: WavelengthGrid, caplog: pytest.LogCaptureFixture
) -> None:
    values = np.array([BT_PLAUSIBLE_MIN_K - 5.0, 200.0, BT_PLAUSIBLE_MAX_K + 5.0, 220.0, 230.0])
    with caplog.at_level(logging.WARNING):
        Spectrum(wavelengths, values, SpectrumKind.BT, "K")
    assert any("plausible range" in record.message for record in caplog.records)


def test_units_warning_for_unexpected_units(
    wavelengths: WavelengthGrid, caplog: pytest.LogCaptureFixture
) -> None:
    values = np.linspace(0.0, 1.0, 5)
    with caplog.at_level(logging.WARNING):
        Spectrum(wavelengths, values, SpectrumKind.RADIANCE, "foo_units")
    assert any("Unexpected units" in record.message for record in caplog.records)
