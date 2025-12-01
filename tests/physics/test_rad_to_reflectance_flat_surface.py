import numpy as np

from alchemi.physics.rad_reflectance import radiance_to_toa_reflectance, toa_reflectance_to_radiance
from alchemi.types import QuantityKind, Spectrum, WavelengthGrid


def test_flat_reflectance_roundtrip():
    wavelengths = WavelengthGrid(np.linspace(400.0, 500.0, 4))
    radiance_values = np.full(wavelengths.nm.shape, 5.0)
    solar_irradiance = np.full_like(wavelengths.nm, 200.0)

    radiance_spec = Spectrum.from_radiance(wavelengths, radiance_values)
    reflectance = radiance_to_toa_reflectance(radiance_spec, solar_zenith_deg=30.0, earth_sun_distance_au=1.0, solar_irradiance_nm=solar_irradiance)
    assert reflectance.kind == QuantityKind.REFLECTANCE
    assert np.allclose(reflectance.values, reflectance.values[0])

    recovered = toa_reflectance_to_radiance(reflectance, solar_zenith_deg=30.0, earth_sun_distance_au=1.0, solar_irradiance_nm=solar_irradiance)
    assert recovered.kind == QuantityKind.RADIANCE
    assert np.allclose(recovered.values, radiance_spec.values)
