import numpy as np
import pytest

from alchemi.physics.rad_reflectance import (
    radiance_sample_to_toa_reflectance,
    radiance_to_toa_reflectance,
    toa_reflectance_sample_to_radiance,
    toa_reflectance_to_radiance,
)
from alchemi.physics.solar import earth_sun_distance_au, esun_for_sample
from alchemi.spectral.sample import Sample, ViewingGeometry
from alchemi.types import QuantityKind, RadianceUnits, Spectrum, WavelengthGrid


def test_radiance_to_reflectance_matches_equation() -> None:
    wavelengths = WavelengthGrid(np.array([1000.0, 1100.0]))
    radiance_values = np.array([5.0, 6.0])
    esun = np.array([200.0, 250.0])
    d_au = 0.99
    solar_zenith = 30.0

    spectrum = Spectrum.from_radiance(wavelengths, radiance_values, units=RadianceUnits.W_M2_SR_NM)
    reflectance = radiance_to_toa_reflectance(
        spectrum, esun_band=esun, d_au=d_au, solar_zenith_deg=solar_zenith
    )

    expected = (np.pi * radiance_values * d_au**2) / (esun * np.cos(np.deg2rad(solar_zenith)))
    np.testing.assert_allclose(reflectance.values, expected)
    assert reflectance.kind == QuantityKind.REFLECTANCE


def test_round_trip_preserves_radiance() -> None:
    wavelengths = WavelengthGrid(np.linspace(400.0, 500.0, 4))
    radiance_values = np.linspace(1.0, 4.0, 4)
    esun = np.linspace(150.0, 250.0, 4)
    d_au = 1.01
    solar_zenith = 10.0

    spectrum = Spectrum.from_radiance(wavelengths, radiance_values, units=RadianceUnits.W_M2_SR_NM)
    reflectance = radiance_to_toa_reflectance(
        spectrum, esun_band=esun, d_au=d_au, solar_zenith_deg=solar_zenith
    )
    recovered = toa_reflectance_to_radiance(
        reflectance, esun_band=esun, d_au=d_au, solar_zenith_deg=solar_zenith
    )

    np.testing.assert_allclose(recovered.values, radiance_values, atol=1e-12)
    assert recovered.kind == QuantityKind.RADIANCE
    assert reflectance.kind == QuantityKind.REFLECTANCE


def test_sample_conversion_uses_solar_utilities() -> None:
    wavelengths = WavelengthGrid(np.array([1500.0, 1510.0]))
    radiance_values = np.array([10.0, 12.0])
    solar_zenith = 20.0
    acquisition_day = 150

    spectrum = Spectrum.from_radiance(wavelengths, radiance_values, units=RadianceUnits.W_M2_SR_NM)
    sample = Sample(
        spectrum=spectrum,
        sensor_id="synthetic",
        viewing_geometry=ViewingGeometry(
            solar_zenith_deg=solar_zenith,
            solar_azimuth_deg=0.0,
            view_zenith_deg=0.0,
            view_azimuth_deg=0.0,
            earth_sun_distance_au=earth_sun_distance_au(doy=acquisition_day),
        ),
        acquisition_time=np.datetime64(f"2024-01-01") + np.timedelta64(acquisition_day - 1, "D"),
    )

    esun_band = esun_for_sample(sample)
    expected = radiance_to_toa_reflectance(
        spectrum,
        esun_band=esun_band,
        d_au=sample.viewing_geometry.earth_sun_distance_au,
        solar_zenith_deg=solar_zenith,
    )

    converted = radiance_sample_to_toa_reflectance(sample)
    np.testing.assert_allclose(converted.spectrum.values, expected.values)
    assert converted.spectrum.kind == QuantityKind.REFLECTANCE
    assert converted.sensor_id == sample.sensor_id


def test_high_reflectance_triggers_warning(caplog: pytest.LogCaptureFixture) -> None:
    wavelengths = WavelengthGrid(np.array([1000.0, 1010.0, 1020.0]))
    radiance_values = np.full(3, 1000.0)
    esun_band = np.full(3, 10.0)

    spectrum = Spectrum.from_radiance(wavelengths, radiance_values, units=RadianceUnits.W_M2_SR_NM)
    with caplog.at_level("WARNING"):
        _ = radiance_to_toa_reflectance(
            spectrum, esun_band=esun_band, d_au=1.0, solar_zenith_deg=5.0
        )

    assert any("High TOA reflectance fraction" in record.message for record in caplog.records)


def test_sample_round_trip() -> None:
    wavelengths = WavelengthGrid(np.array([2000.0, 2010.0]))
    radiance_values = np.array([3.0, 4.0])
    solar_zenith = 25.0

    spectrum = Spectrum.from_radiance(wavelengths, radiance_values, units=RadianceUnits.W_M2_SR_NM)
    sample = Sample(
        spectrum=spectrum,
        sensor_id="synthetic",
        viewing_geometry=ViewingGeometry(
            solar_zenith_deg=solar_zenith,
            solar_azimuth_deg=0.0,
            view_zenith_deg=0.0,
            view_azimuth_deg=0.0,
            earth_sun_distance_au=1.0,
        ),
    )

    esun_band = np.array([180.0, 190.0])
    to_reflectance = radiance_sample_to_toa_reflectance(
        sample, esun_ref=Spectrum(wavelength_nm=wavelengths.nm, values=esun_band, kind=QuantityKind.RADIANCE)
    )
    back_to_radiance = toa_reflectance_sample_to_radiance(
        to_reflectance,
        esun_ref=Spectrum(wavelength_nm=wavelengths.nm, values=esun_band, kind=QuantityKind.RADIANCE),
    )

    np.testing.assert_allclose(back_to_radiance.spectrum.values, radiance_values)
    assert back_to_radiance.spectrum.kind == QuantityKind.RADIANCE
