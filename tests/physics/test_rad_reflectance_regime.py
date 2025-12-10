import logging

import numpy as np
import pytest

from alchemi.physics.rad_reflectance import (
    radiance_sample_to_toa_reflectance,
    radiance_to_toa_reflectance,
)
from alchemi.physics.rt_regime import SWIRRegime
from alchemi.spectral.sample import Sample, ViewingGeometry
from alchemi.types import RadianceUnits, Spectrum, WavelengthGrid


def _simple_radiance_spectrum() -> Spectrum:
    wavelengths = WavelengthGrid(np.array([1000.0, 1010.0], dtype=np.float64))
    return Spectrum.from_radiance(wavelengths, np.array([1.0, 1.2], dtype=np.float64))


def _simple_sample(regime: SWIRRegime) -> Sample:
    spectrum = _simple_radiance_spectrum()
    geometry = ViewingGeometry(30.0, 0.0, 0.0, 0.0, 1.0)
    ancillary = {"swir_regime": regime.value}
    return Sample(spectrum=spectrum, sensor_id="sensor", viewing_geometry=geometry, ancillary=ancillary)


def _esun_for_test() -> Spectrum:
    spectrum = _simple_radiance_spectrum()
    return Spectrum(
        wavelengths=spectrum.wavelengths,
        values=np.full_like(spectrum.values, 150.0),
        kind=spectrum.kind,
        units=RadianceUnits.W_M2_SR_NM,
    )


def test_radiance_to_toa_reflectance_warns_under_heavy_regime(caplog: pytest.LogCaptureFixture):
    spectrum = _simple_radiance_spectrum()
    esun_band = np.full_like(spectrum.wavelengths.nm, 120.0)

    with caplog.at_level(logging.WARNING):
        radiance_to_toa_reflectance(
            spectrum,
            esun_band=esun_band,
            d_au=1.0,
            solar_zenith_deg=30.0,
            swir_regime=SWIRRegime.HEAVY,
        )

    assert any("heavy" in record.message for record in caplog.records)


def test_sample_conversion_can_warn_and_be_suppressed(caplog: pytest.LogCaptureFixture):
    sample = _simple_sample(SWIRRegime.HEAVY)
    esun_ref = _esun_for_test()

    with caplog.at_level(logging.WARNING):
        radiance_sample_to_toa_reflectance(sample, esun_ref=esun_ref)

    assert any("heavy" in record.message for record in caplog.records)

    caplog.clear()

    with caplog.at_level(logging.WARNING):
        radiance_sample_to_toa_reflectance(sample, esun_ref=esun_ref, warn_outside_trusted=False)

    assert not any("heavy" in record.message for record in caplog.records)
