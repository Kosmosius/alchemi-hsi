import numpy as np
import pytest

from alchemi.data.io.hytes import HYTES_WAVELENGTHS_NM
from alchemi.physics.planck import (
    bt_K_to_radiance,
    inverse_planck_central_lambda,
    planck_radiance_wavelength,
    radiance_to_bt_K,
)


def test_hytes_roundtrip_temperatures():
    temps = np.array([200.0, 240.0, 280.0, 320.0])
    radiance = planck_radiance_wavelength(HYTES_WAVELENGTHS_NM, temps[:, None])
    recovered = radiance_to_bt_K(radiance, HYTES_WAVELENGTHS_NM)
    expected = np.broadcast_to(temps[:, None], recovered.shape)
    np.testing.assert_allclose(recovered, expected, atol=5e-2)


def test_vectorized_planck_scalar_temperature_vector_wavelength():
    wl_nm = np.linspace(8_000.0, 12_000.0, 16)
    temp = 305.0
    radiance = planck_radiance_wavelength(wl_nm, temp)
    recovered = radiance_to_bt_K(radiance, wl_nm)
    assert radiance.shape == wl_nm.shape
    np.testing.assert_allclose(recovered, temp, atol=1e-2)


def test_vectorized_planck_vector_temperature_scalar_wavelength():
    wl_nm = 10_000.0
    temps = np.linspace(220.0, 340.0, 5)
    radiance = planck_radiance_wavelength(wl_nm, temps)
    recovered = radiance_to_bt_K(radiance, wl_nm)
    assert radiance.shape == temps.shape
    np.testing.assert_allclose(recovered, temps, atol=1e-2)


def test_broadcasted_planck_shapes():
    wl_nm = np.linspace(8_000.0, 12_000.0, 8)
    temps = np.array([[250.0], [280.0], [310.0]])
    radiance = planck_radiance_wavelength(wl_nm, temps)
    recovered = radiance_to_bt_K(radiance, wl_nm)
    assert radiance.shape == (temps.shape[0], wl_nm.shape[0])
    expected = np.broadcast_to(temps, recovered.shape)
    np.testing.assert_allclose(recovered, expected, atol=1e-2)


def test_radiance_shape_and_peak_behavior():
    wl_nm = HYTES_WAVELENGTHS_NM
    temp = 300.0
    radiance = planck_radiance_wavelength(wl_nm, temp)

    peak_idx = int(np.argmax(radiance))
    peak_wl = wl_nm[peak_idx]

    assert 9_000.0 <= peak_wl <= 11_000.0
    assert radiance[peak_idx] > radiance[0]
    assert radiance[peak_idx] > radiance[-1]

    left = radiance[: peak_idx + 1]
    right = radiance[peak_idx:]
    np.testing.assert_allclose(np.diff(left) >= 0.0, True)
    np.testing.assert_allclose(np.diff(right) <= 0.0, True)


def test_nm_and_micron_scaling_equivalence():
    wl_nm = np.array([8_500.0, 10_000.0, 11_500.0])
    temp = 295.0
    radiance_per_nm = planck_radiance_wavelength(wl_nm, temp)

    wl_um = wl_nm * 1e-3
    radiance_per_um = planck_radiance_wavelength(wl_um * 1e3, temp) * 1e3

    np.testing.assert_allclose(radiance_per_um / 1e3, radiance_per_nm, rtol=1e-12, atol=0.0)


def test_invalid_inputs_raise_errors():
    with pytest.raises(ValueError):
        planck_radiance_wavelength(-5.0, 300.0)
    with pytest.raises(ValueError):
        planck_radiance_wavelength(10_000.0, 0.0)
    with pytest.raises(ValueError):
        inverse_planck_central_lambda(-1.0, 10_000.0)


def test_non_positive_radiance_yields_zero_bt():
    wl_nm = np.array([9_000.0, 10_000.0])
    radiance = np.array([0.0, -1.0])
    recovered = radiance_to_bt_K(radiance, wl_nm)
    np.testing.assert_array_equal(recovered, np.zeros_like(radiance))
