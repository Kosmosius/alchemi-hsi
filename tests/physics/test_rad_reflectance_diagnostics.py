import logging

import numpy as np
import pytest

from alchemi.physics.rad_reflectance import _diagnose_reflectance, radiance_to_toa_reflectance
from alchemi.types import Spectrum, WavelengthGrid


def test_diagnose_reflectance_warns_and_reports_range(caplog: pytest.LogCaptureFixture):
    reflectance_values = np.array([0.8, 1.25, 1.6, 2.2, 0.9], dtype=np.float64)

    with caplog.at_level(logging.WARNING):
        _diagnose_reflectance(reflectance_values)

    assert any("diagnostics" in record.message for record in caplog.records)
    assert any("range" in record.message for record in caplog.records)


def test_diagnose_reflectance_can_be_strict():
    reflectance_values = np.array([0.8, 1.25, 1.6, 2.2, 0.9], dtype=np.float64)

    with pytest.raises(ValueError, match="TOA reflectance diagnostics"):
        _diagnose_reflectance(reflectance_values, strict=True)


def test_invalid_inputs_raise_clear_errors():
    wavelengths = WavelengthGrid(np.linspace(400.0, 500.0, 3))
    radiance_values = np.full(wavelengths.nm.shape, 3.0)

    radiance_spec = Spectrum.from_radiance(wavelengths, radiance_values)

    with pytest.raises(ValueError, match="Solar irradiance"):
        radiance_to_toa_reflectance(
            radiance_spec,
            esun_band=np.array([100.0, -50.0, 120.0]),
            d_au=1.0,
            solar_zenith_deg=30.0,
        )

    with pytest.raises(ValueError, match="Earth-Sun distance"):
        radiance_to_toa_reflectance(
            radiance_spec,
            esun_band=np.full_like(wavelengths.nm, 100.0),
            d_au=0.0,
            solar_zenith_deg=30.0,
        )

    with pytest.raises(ValueError, match="Cosine of solar zenith"):
        radiance_to_toa_reflectance(
            radiance_spec,
            esun_band=np.full_like(wavelengths.nm, 100.0),
            d_au=1.0,
            solar_zenith_deg=90.0,
        )
