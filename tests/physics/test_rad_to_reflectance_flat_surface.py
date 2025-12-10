import numpy as np
import pytest

from alchemi.physics.rad_reflectance import radiance_to_toa_reflectance, toa_reflectance_to_radiance
from alchemi.types import QuantityKind, Spectrum, WavelengthGrid


def test_flat_reflectance_roundtrip():
    wavelengths = WavelengthGrid(np.linspace(400.0, 500.0, 4))
    radiance_values = np.full(wavelengths.nm.shape, 5.0)
    solar_irradiance = np.full_like(wavelengths.nm, 200.0)

    radiance_spec = Spectrum.from_radiance(wavelengths, radiance_values)
    reflectance = radiance_to_toa_reflectance(
        radiance_spec,
        esun_band=solar_irradiance,
        d_au=1.0,
        solar_zenith_deg=30.0,
    )
    assert reflectance.kind == QuantityKind.TOA_REFLECTANCE
    assert np.allclose(reflectance.values, reflectance.values[0])

    recovered = toa_reflectance_to_radiance(
        reflectance,
        esun_band=solar_irradiance,
        d_au=1.0,
        solar_zenith_deg=30.0,
    )
    assert recovered.kind == QuantityKind.RADIANCE
    assert np.allclose(recovered.values, radiance_spec.values)


def test_surface_reflectance_rejected_for_toa_inverse():
    wavelengths = WavelengthGrid(np.linspace(400.0, 500.0, 4))
    reflectance_values = np.full(wavelengths.nm.shape, 0.1)
    solar_irradiance = np.full_like(wavelengths.nm, 200.0)

    surface = Spectrum.from_surface_reflectance(wavelengths, reflectance_values)

    with pytest.raises(ValueError, match="TOA reflectance"):
        toa_reflectance_to_radiance(
            surface,
            esun_band=solar_irradiance,
            d_au=1.0,
            solar_zenith_deg=30.0,
        )
