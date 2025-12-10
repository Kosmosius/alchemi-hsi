import numpy as np
import pytest

from alchemi.physics import units
from alchemi.physics.planck import planck_radiance_wavelength, radiance_spectrum_to_bt_central
from alchemi.physics.rad_reflectance import radiance_to_toa_reflectance
from alchemi.types import RadianceUnits, Spectrum, WavelengthGrid

pytestmark = pytest.mark.physics_and_metadata


def test_radiance_roundtrip_nm_um_and_wavenumber():
    wavelengths_nm = np.array([450.0, 600.0, 900.0, 1_200.0])
    radiance_nm = np.array([1.5, 2.0, 4.0, 5.0], dtype=np.float64)

    per_um = units.scale_radiance_between_wavelength_units(
        radiance_nm, units.ValueUnits.RADIANCE_W_M2_SR_NM, units.ValueUnits.RADIANCE_W_M2_SR_UM
    )
    back_to_nm = units.scale_radiance_between_wavelength_units(
        per_um, units.ValueUnits.RADIANCE_W_M2_SR_UM, units.ValueUnits.RADIANCE_W_M2_SR_NM
    )

    wavenumber_cm1, radiance_per_cm1 = units.radiance_wavelength_nm_to_wavenumber_cm1(
        back_to_nm, wavelengths_nm
    )
    restored_wavelengths, restored_radiance_nm = units.radiance_wavenumber_cm1_to_wavelength_nm(
        radiance_per_cm1, wavenumber_cm1
    )

    np.testing.assert_allclose(back_to_nm, radiance_nm, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(restored_radiance_nm, radiance_nm, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(restored_wavelengths, wavelengths_nm, rtol=0.0, atol=1e-12)


def test_planck_bt_matches_after_unit_normalization():
    wavelengths_nm = np.array([8_500.0, 9_750.0, 11_250.0])
    radiance_nm = planck_radiance_wavelength(wavelengths_nm, 310.0)
    radiance_um = units.scale_radiance_between_wavelength_units(
        radiance_nm, units.ValueUnits.RADIANCE_W_M2_SR_NM, units.ValueUnits.RADIANCE_W_M2_SR_UM
    )

    spectrum_nm = Spectrum.from_radiance(
        WavelengthGrid(wavelengths_nm), radiance_nm, units=RadianceUnits.W_M2_SR_NM
    )
    spectrum_um = Spectrum.from_radiance(
        WavelengthGrid(wavelengths_nm), radiance_um, units=RadianceUnits.W_M2_SR_UM
    )

    bt_nm = radiance_spectrum_to_bt_central(spectrum_nm)
    bt_um = radiance_spectrum_to_bt_central(spectrum_um)

    np.testing.assert_allclose(bt_nm.values, bt_um.values, atol=5e-6)


def test_reflectance_is_invariant_to_radiance_units():
    wavelengths_nm = np.array([430.0, 660.0, 860.0])
    radiance_nm = np.array([10.0, 12.0, 8.5], dtype=np.float64)
    esun = np.array([1_600.0, 1_550.0, 1_400.0], dtype=np.float64)

    spectrum_nm = Spectrum.from_radiance(
        WavelengthGrid(wavelengths_nm), radiance_nm, units=RadianceUnits.W_M2_SR_NM
    )
    spectrum_um = Spectrum.from_radiance(
        WavelengthGrid(wavelengths_nm),
        units.scale_radiance_between_wavelength_units(
            radiance_nm, units.ValueUnits.RADIANCE_W_M2_SR_NM, units.ValueUnits.RADIANCE_W_M2_SR_UM
        ),
        units=RadianceUnits.W_M2_SR_UM,
    )

    refl_nm = radiance_to_toa_reflectance(
        spectrum_nm, esun_band=esun, d_au=1.0, solar_zenith_deg=30.0
    )
    refl_um = radiance_to_toa_reflectance(
        spectrum_um, esun_band=esun, d_au=1.0, solar_zenith_deg=30.0
    )

    np.testing.assert_allclose(refl_nm.values, refl_um.values, rtol=0.0, atol=1e-10)
