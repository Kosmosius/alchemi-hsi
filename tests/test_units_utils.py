import numpy as np

from alchemi.physics import units as qty_units
from alchemi.types import QuantityKind, Spectrum, ValueUnits, WavelengthGrid


def test_radiance_scaling_um_to_nm():
    values = np.array([1.0, 2.0])
    scaled = qty_units.scale_radiance_between_wavelength_units(
        values, ValueUnits.RADIANCE_W_M2_SR_UM, ValueUnits.RADIANCE_W_M2_SR_NM
    )
    assert np.allclose(scaled, values / 1000.0)


def test_wavelength_conversions():
    nm = qty_units.wavelength_um_to_nm(np.array([1.0, 2.5]))
    assert np.allclose(nm, np.array([1000.0, 2500.0]))
    um = qty_units.wavelength_nm_to_um(np.array([1000.0]))
    assert np.allclose(um, np.array([1.0]))


def test_wavenumber_to_wavelength_nm():
    wl_nm = qty_units.wavenumber_cm1_to_wavelength_nm(np.array([10000.0]))
    assert np.allclose(wl_nm, np.array([1000.0]))


def test_spectrum_normalizes_units():
    wavelengths = WavelengthGrid(np.array([400.0, 500.0]))
    spec = Spectrum(
        wavelengths=wavelengths,
        values=np.array([10.0, 20.0]),
        kind=QuantityKind.RADIANCE,
        units=ValueUnits.RADIANCE_W_M2_SR_UM,
    )
    assert spec.units == ValueUnits.RADIANCE_W_M2_SR_NM
    assert np.allclose(spec.values, np.array([0.01, 0.02]))


def test_reflectance_percent_normalized():
    wavelengths = WavelengthGrid(np.array([400.0]))
    spec = Spectrum(
        wavelengths=wavelengths,
        values=np.array([50.0]),
        kind=QuantityKind.REFLECTANCE,
        units=ValueUnits.REFLECTANCE_PERCENT,
    )
    assert spec.units == ValueUnits.REFLECTANCE_FRACTION
    assert np.allclose(spec.values, np.array([0.5]))


def test_temperature_celsius_normalized():
    wavelengths = WavelengthGrid(np.array([400.0]))
    spec = Spectrum(
        wavelengths=wavelengths,
        values=np.array([0.0]),
        kind=QuantityKind.BRIGHTNESS_T,
        units=ValueUnits.TEMPERATURE_C,
    )
    assert spec.units == ValueUnits.TEMPERATURE_K
    assert np.isclose(spec.values[0], 273.15)
