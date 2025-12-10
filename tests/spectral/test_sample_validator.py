import numpy as np
import pytest

from alchemi.registry.sensors import SensorSpec
from alchemi.spectral import BandMetadata, Sample, Spectrum, ViewingGeometry
from alchemi.spectral.validate import validate_sample, validate_spectrum_physics
from alchemi.types import QuantityKind, RadianceUnits, ReflectanceUnits, TemperatureUnits


@pytest.fixture
def base_wavelengths():
    return np.array([400.0, 500.0, 600.0], dtype=np.float64)


def test_validate_spectrum_physics_all_kinds(base_wavelengths):
    radiance = Spectrum(
        wavelength_nm=base_wavelengths,
        values=np.full(3, 1.0, dtype=np.float64),
        kind=QuantityKind.RADIANCE,
        units=RadianceUnits.W_M2_SR_NM,
    )
    validate_spectrum_physics(radiance)

    reflectance = Spectrum(
        wavelength_nm=base_wavelengths,
        values=np.full(3, 0.2, dtype=np.float64),
        kind=QuantityKind.REFLECTANCE,
        units=ReflectanceUnits.FRACTION,
    )
    validate_spectrum_physics(reflectance)

    bt = Spectrum(
        wavelength_nm=base_wavelengths,
        values=np.full(3, 300.0, dtype=np.float64),
        kind=QuantityKind.BRIGHTNESS_T,
        units=TemperatureUnits.KELVIN,
    )
    validate_spectrum_physics(bt)


def test_validate_spectrum_monotonicity_failure(base_wavelengths):
    spectrum = Spectrum(
        wavelength_nm=base_wavelengths,
        values=np.full(3, 1.0, dtype=np.float64),
        kind=QuantityKind.RADIANCE,
        units=RadianceUnits.W_M2_SR_NM,
    )
    spectrum.wavelengths.nm[1] = spectrum.wavelengths.nm[0]

    with pytest.raises(ValueError):
        validate_spectrum_physics(spectrum)


def test_validate_spectrum_radiance_units_failure(base_wavelengths):
    spectrum = Spectrum(
        wavelength_nm=base_wavelengths,
        values=np.full(3, 1.0, dtype=np.float64),
        kind=QuantityKind.RADIANCE,
        units=RadianceUnits.W_M2_SR_NM,
    )
    spectrum.units = ReflectanceUnits.FRACTION

    with pytest.raises(ValueError):
        validate_spectrum_physics(spectrum)


def test_validate_spectrum_reflectance_range_failure(base_wavelengths):
    spectrum = Spectrum(
        wavelength_nm=base_wavelengths,
        values=np.full(3, 0.2, dtype=np.float64),
        kind=QuantityKind.REFLECTANCE,
        units=ReflectanceUnits.FRACTION,
    )
    spectrum.values[:] = [10.0, -0.5, 0.1]

    with pytest.raises(ValueError):
        validate_spectrum_physics(spectrum)


def make_sensor_spec(expected_band_count: int, wavelength_min: float, wavelength_max: float):
    centers = np.linspace(wavelength_min, wavelength_max, expected_band_count, dtype=np.float64)
    widths = np.full_like(centers, 10.0)
    return SensorSpec(
        sensor_id="test-sensor",
        expected_band_count=expected_band_count,
        wavelength_range_nm=(wavelength_min, wavelength_max),
        band_centers_nm=centers,
        band_widths_nm=widths,
        srf_source="official",
    )


def test_validate_sample_with_sensor_spec(base_wavelengths):
    sensor_spec = make_sensor_spec(3, 380.0, 620.0)
    band_meta = BandMetadata(center_nm=base_wavelengths, width_nm=None, valid_mask=np.ones(3, dtype=bool))
    sample = Sample(
        spectrum=Spectrum(
            wavelength_nm=base_wavelengths,
            values=np.full(3, 1.0, dtype=np.float64),
            kind=QuantityKind.RADIANCE,
            units=RadianceUnits.W_M2_SR_NM,
        ),
        sensor_id=sensor_spec.sensor_id,
        band_meta=band_meta,
    )

    validate_sample(sample, sensor_spec)


def test_validate_sample_wavelength_out_of_range(base_wavelengths):
    sensor_spec = make_sensor_spec(3, 400.0, 700.0)
    bad_wavelengths = base_wavelengths + 500.0
    sample = Sample(
        spectrum=Spectrum(
            wavelength_nm=bad_wavelengths,
            values=np.full(3, 0.2, dtype=np.float64),
            kind=QuantityKind.REFLECTANCE,
            units=ReflectanceUnits.FRACTION,
        ),
        sensor_id=sensor_spec.sensor_id,
    )

    with pytest.raises(ValueError):
        validate_sample(sample, sensor_spec)


def test_validate_sample_band_count_mismatch(base_wavelengths):
    sensor_spec = make_sensor_spec(20, 380.0, 620.0)
    sample = Sample(
        spectrum=Spectrum(
            wavelength_nm=base_wavelengths,
            values=np.full(3, 1.0, dtype=np.float64),
            kind=QuantityKind.RADIANCE,
            units=RadianceUnits.W_M2_SR_NM,
        ),
        sensor_id=sensor_spec.sensor_id,
    )

    with pytest.raises(ValueError):
        validate_sample(sample, sensor_spec)


def test_validate_sample_band_centers_mismatch(base_wavelengths):
    sensor_spec = make_sensor_spec(3, 380.0, 620.0)
    shifted_centers = base_wavelengths + 50.0
    band_meta = BandMetadata(center_nm=shifted_centers, width_nm=None, valid_mask=np.ones(3, dtype=bool))
    sample = Sample(
        spectrum=Spectrum(
            wavelength_nm=base_wavelengths,
            values=np.full(3, 0.2, dtype=np.float64),
            kind=QuantityKind.REFLECTANCE,
            units=ReflectanceUnits.FRACTION,
        ),
        sensor_id=sensor_spec.sensor_id,
        band_meta=band_meta,
    )

    with pytest.raises(ValueError):
        validate_sample(sample, sensor_spec)


def test_validate_sample_radiance_geometry_check(base_wavelengths):
    sensor_spec = make_sensor_spec(3, 380.0, 620.0)
    sample = Sample(
        spectrum=Spectrum(
            wavelength_nm=base_wavelengths,
            values=np.full(3, 1.0, dtype=np.float64),
            kind=QuantityKind.RADIANCE,
            units=RadianceUnits.W_M2_SR_NM,
        ),
        sensor_id=sensor_spec.sensor_id,
        viewing_geometry=ViewingGeometry(
            solar_zenith_deg=95.0,
            solar_azimuth_deg=0.0,
            view_zenith_deg=0.0,
            view_azimuth_deg=0.0,
            earth_sun_distance_au=1.0,
        ),
    )

    with pytest.raises(ValueError):
        validate_sample(sample, sensor_spec)
