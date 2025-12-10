import numpy as np

from alchemi.physics import planck, tes
from alchemi.types import QuantityKind, RadianceUnits, Sample, Spectrum, WavelengthGrid


def _blackbody_radiance_stack(wavelengths_nm: np.ndarray, temperatures_K: np.ndarray) -> np.ndarray:
    return np.stack(
        [planck.planck_radiance_wavelength(wavelengths_nm, temp) for temp in temperatures_K],
        axis=0,
    )


def test_blackbody_emissivity_proxy_recovers_temperature() -> None:
    wavelengths = WavelengthGrid(np.linspace(8000.0, 12000.0, 6))
    true_temperature = 315.0

    radiance = planck.planck_radiance_wavelength(wavelengths.nm, true_temperature)
    spectrum = Spectrum.from_radiance(wavelengths, radiance, units=RadianceUnits.W_M2_SR_NM)
    sample = Sample(spectrum=spectrum, sensor_id="synthetic")

    result = tes.lwir_pipeline_for_sample(sample)

    T_proxy = result.ancillary["lwir_T_proxy_K"]
    emissivity_proxy: Spectrum = result.ancillary["lwir_emissivity_proxy"]

    assert np.isclose(T_proxy, true_temperature, atol=1e-3)
    assert emissivity_proxy.kind == QuantityKind.SURFACE_REFLECTANCE
    assert np.allclose(emissivity_proxy.values, 1.0, atol=1e-6)


def test_emissivity_proxy_vectorised_pixels() -> None:
    wavelengths = WavelengthGrid(np.linspace(8000.0, 12000.0, 8))
    temperatures = np.array([290.0, 320.0])

    radiance = _blackbody_radiance_stack(wavelengths.nm, temperatures)
    spectrum = Spectrum.from_radiance(wavelengths, radiance, units=RadianceUnits.W_M2_SR_NM)
    sample = Sample(spectrum=spectrum, sensor_id="synthetic_stack")

    result = tes.lwir_pipeline_for_sample(sample)

    T_proxy = result.ancillary["lwir_T_proxy_K"]
    emissivity_proxy: Spectrum = result.ancillary["lwir_emissivity_proxy"]

    assert np.allclose(T_proxy, temperatures, atol=1e-3)
    assert emissivity_proxy.values.shape == radiance.shape
    assert np.allclose(emissivity_proxy.values, 1.0, atol=1e-6)
