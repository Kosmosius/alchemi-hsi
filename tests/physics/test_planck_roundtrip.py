import numpy as np
import pytest

from alchemi.data.io.hytes import HYTES_WAVELENGTHS_NM
from alchemi.physics import units
from alchemi.physics.planck import (
    band_averaged_radiance,
    hytes_band_averaged_radiance_to_bt,
    inverse_planck_central_lambda,
    invert_band_averaged_radiance_to_bt,
    planck_radiance_wavelength,
    radiance_to_bt_K,
)
from alchemi.srf.hytes import hytes_srf_matrix
from alchemi.srf.registry import sensor_srf_from_legacy

pytestmark = pytest.mark.physics_and_metadata


def test_central_wavelength_roundtrip_precision():
    wavelengths_nm = np.array([8_300.0, 9_500.0, 10_700.0, 11_900.0])
    temps = np.linspace(250.0, 330.0, 9)

    radiance = planck_radiance_wavelength(wavelengths_nm, temps[:, None])
    recovered_bt = inverse_planck_central_lambda(radiance, wavelengths_nm)
    reconstructed_radiance = planck_radiance_wavelength(wavelengths_nm, recovered_bt)

    reshape = temps.reshape((-1,) + (1,) * (recovered_bt.ndim - 1))
    expected_bt = np.broadcast_to(reshape, recovered_bt.shape)
    np.testing.assert_allclose(recovered_bt, expected_bt, atol=5e-3)
    np.testing.assert_allclose(reconstructed_radiance, radiance, rtol=1e-6)


def test_band_averaged_roundtrip_hytes_like():
    srf_legacy = hytes_srf_matrix()
    grid = np.linspace(float(srf_legacy.bands_nm[0][0]), float(srf_legacy.bands_nm[-1][-1]), 1024)
    srf_dense = sensor_srf_from_legacy(hytes_srf_matrix(), grid=grid).as_matrix()
    srf_dense.normalize_rows_trapz()
    temps = np.linspace(260.0, 330.0, 6)

    band_radiance = band_averaged_radiance(temps, srf_dense.matrix, srf_dense.wavelength_nm)

    recovered_bt = invert_band_averaged_radiance_to_bt(
        band_radiance,
        srf_dense.matrix,
        srf_dense.wavelength_nm,
        temps_grid_K=np.arange(200.0, 360.0, 0.25),
    )
    reconstructed = band_averaged_radiance(recovered_bt[:, 0], srf_dense.matrix, srf_dense.wavelength_nm)

    reshape = temps.reshape((-1,) + (1,) * (recovered_bt.ndim - 1))
    expected_bt = np.broadcast_to(reshape, recovered_bt.shape)
    np.testing.assert_allclose(recovered_bt, expected_bt, atol=5e-2)
    np.testing.assert_allclose(reconstructed, band_radiance, rtol=5e-4)

    hytes_bt = hytes_band_averaged_radiance_to_bt(band_radiance, sensor_srf=srf_dense)
    np.testing.assert_allclose(hytes_bt, expected_bt, atol=5e-2)


def test_monotonicity():
    wl_nm = HYTES_WAVELENGTHS_NM[::32]
    temps = np.linspace(210.0, 340.0, 12)
    radiance = planck_radiance_wavelength(wl_nm, temps[:, None])

    diffs = np.diff(radiance, axis=0)
    assert np.all(diffs > 0)

    wl_single = 10_000.0
    L = planck_radiance_wavelength(wl_single, temps)
    recovered = inverse_planck_central_lambda(L, wl_single)
    assert np.all(np.diff(recovered) > 0)


def test_unit_consistency_and_scaling():
    wl_nm = np.array([8_500.0, 9_750.0, 11_250.0])
    temp = 305.0

    radiance_per_nm = planck_radiance_wavelength(wl_nm, temp)
    bt_nm = radiance_to_bt_K(radiance_per_nm, wl_nm)

    radiance_per_um = units.scale_radiance_between_wavelength_units(
        radiance_per_nm, units.ValueUnits.RADIANCE_W_M2_SR_NM, units.ValueUnits.RADIANCE_W_M2_SR_UM
    )
    bt_from_um = radiance_to_bt_K(
        units.scale_radiance_between_wavelength_units(
            radiance_per_um, units.ValueUnits.RADIANCE_W_M2_SR_UM, units.ValueUnits.RADIANCE_W_M2_SR_NM
        ),
        wl_nm,
    )

    np.testing.assert_allclose(bt_nm, bt_from_um, atol=1e-8)

    wl_m = wl_nm * 1e-9
    radiance_from_m = planck_radiance_wavelength(wl_m * 1e9, temp)
    np.testing.assert_allclose(radiance_from_m, radiance_per_nm, rtol=1e-12)


def test_vector_shapes_and_radiance_to_bt_back():
    wl_nm = np.linspace(8_000.0, 12_000.0, 32)
    temps = np.array([[260.0], [290.0], [320.0]])

    radiance = planck_radiance_wavelength(wl_nm, temps)
    recovered = radiance_to_bt_K(radiance, wl_nm)
    np.testing.assert_allclose(recovered, np.broadcast_to(temps, recovered.shape), atol=1e-2)

