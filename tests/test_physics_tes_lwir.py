"""Tests for LWIR brightness-temperature and emissivity-proxy utilities."""

from __future__ import annotations

import numpy as np
import pytest

from alchemi.physics.planck import planck_radiance_wavelength
from alchemi.physics.tes import (
    bt_spectrum_to_radiance_spectrum,
    compute_lwir_emissivity_proxy,
    lwir_pipeline_for_sample,
    radiance_spectrum_to_bt_spectrum,
)
from alchemi.spectral import Sample
from alchemi.types import QuantityKind, Spectrum, TemperatureUnits


def test_radiance_bt_roundtrip() -> None:
    wavelengths = np.linspace(7_500.0, 12_000.0, 8)
    temps = np.linspace(280.0, 320.0, wavelengths.size)
    radiance = planck_radiance_wavelength(wavelengths, temps)
    spectrum = Spectrum.from_radiance(
        wavelengths=wavelengths, values=radiance, units="W·m⁻²·sr⁻¹·nm⁻¹"
    )

    bt_spectrum = radiance_spectrum_to_bt_spectrum(spectrum)
    np.testing.assert_allclose(bt_spectrum.values, temps, rtol=1e-5, atol=1e-3)
    assert bt_spectrum.kind == QuantityKind.BRIGHTNESS_T
    assert bt_spectrum.units == TemperatureUnits.KELVIN.value

    radiance_roundtrip = bt_spectrum_to_radiance_spectrum(bt_spectrum)
    np.testing.assert_allclose(radiance_roundtrip.values, radiance, rtol=1e-6, atol=1e-12)
    assert radiance_roundtrip.kind == QuantityKind.RADIANCE


def test_emissivity_proxy_basic() -> None:
    bt_values = np.array([300.0, 295.0, 290.0])
    wavelengths = np.array([8_000.0, 9_000.0, 10_000.0])
    bt_spectrum = Spectrum.from_brightness_temperature(wavelengths=wavelengths, values=bt_values)

    T_proxy, emissivity_proxy = compute_lwir_emissivity_proxy(bt_spectrum)

    np.testing.assert_allclose(T_proxy, 300.0)
    expected = bt_values / 300.0
    np.testing.assert_allclose(emissivity_proxy.values, expected, rtol=1e-6)
    assert emissivity_proxy.kind == QuantityKind.REFLECTANCE
    assert emissivity_proxy.meta.get("role") == "lwir_emissivity_proxy"
    assert np.all(emissivity_proxy.values <= 1.0 + 1e-6)
    assert np.all(emissivity_proxy.values >= 0.0)


def test_emissivity_proxy_all_nan_raises() -> None:
    wavelengths = np.array([8_000.0, 9_000.0, 10_000.0])
    bt_values = np.array([np.nan, np.nan, np.nan])
    bt_spectrum = Spectrum.from_brightness_temperature(wavelengths=wavelengths, values=bt_values)

    with pytest.raises(ValueError):
        compute_lwir_emissivity_proxy(bt_spectrum)


def test_lwir_pipeline_from_radiance_sample() -> None:
    wavelengths = np.linspace(7_500.0, 12_000.0, 4)
    temps = np.full(wavelengths.shape, 310.0)
    radiance = planck_radiance_wavelength(wavelengths, temps)
    spectrum = Spectrum.from_radiance(wavelengths=wavelengths, values=radiance)
    sample = Sample(spectrum=spectrum, sensor_id="hytes")

    outputs = lwir_pipeline_for_sample(sample)
    assert set(outputs.keys()) == {"radiance", "bt", "emissivity_proxy", "T_proxy"}
    assert outputs["radiance"] is spectrum
    assert outputs["bt"].kind == QuantityKind.BRIGHTNESS_T
    assert outputs["emissivity_proxy"].kind == QuantityKind.REFLECTANCE
    np.testing.assert_allclose(outputs["T_proxy"], temps.max())
