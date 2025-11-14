from __future__ import annotations

import numpy as np
import pytest

from alchemi.physics.swir_enmap import (
    interpolate_irradiance_to_bands,
    radiance_to_reflectance_enmap,
    reflectance_to_radiance_enmap,
)


def test_identity_roundtrip():
    bands = 224
    wl_nm = np.linspace(420.0, 2450.0, bands, dtype=np.float64)
    wl_E0 = np.linspace(400.0, 2500.0, 5000, dtype=np.float64)
    E0_hi = 1500.0 + 0.2 * wl_E0 + 10.0 * np.sin(wl_E0 / 45.0)
    E0_bands = interpolate_irradiance_to_bands(wl_nm, wl_E0, E0_hi)

    R = 0.2 + 0.3 * np.sin(wl_nm / 120.0) ** 2
    L = reflectance_to_radiance_enmap(R, wl_nm, E0_bands, cos_sun=1.0)
    R_back = radiance_to_reflectance_enmap(L, wl_nm, E0_bands, cos_sun=1.0)

    np.testing.assert_allclose(R_back, R, atol=1e-6)


def test_interpolate_irradiance_to_bands_linear():
    wl_nm = np.linspace(420.0, 2450.0, 200)
    wl_E0 = np.linspace(400.0, 2500.0, 40)

    def analytic(wl: np.ndarray) -> np.ndarray:
        return 1000.0 + 0.1 * wl

    E0 = analytic(wl_E0)
    interpolated = interpolate_irradiance_to_bands(wl_nm, wl_E0, E0)
    expected = analytic(wl_nm)

    rel_error = np.abs((interpolated - expected) / expected)
    assert np.max(rel_error) < 1e-3


def test_band_mask_behavior():
    bands = 224
    wl_nm = np.linspace(420.0, 2450.0, bands)
    E0 = np.linspace(1500.0, 1600.0, bands)
    R = np.linspace(0.05, 0.25, bands)
    mask = np.ones(bands, dtype=bool)
    mask[[10, 20, 30]] = False

    L = reflectance_to_radiance_enmap(R, wl_nm, E0, cos_sun=0.8, band_mask=mask)
    assert np.all(np.isnan(L[~mask]))
    assert np.all(np.isfinite(L[mask]))

    R_back = radiance_to_reflectance_enmap(L, wl_nm, E0, cos_sun=0.8, band_mask=mask)
    assert np.all(np.isnan(R_back[~mask]))
    assert np.all(np.isfinite(R_back[mask]))


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_dtype_and_broadcasting(dtype: np.dtype):
    bands = 128
    wl_nm = np.linspace(420.0, 2450.0, bands, dtype=dtype)
    E0 = np.linspace(1300.0, 1700.0, bands, dtype=dtype)
    R = np.linspace(0.1, 0.3, bands, dtype=dtype)[None, :]
    cos_sun = np.array([[0.9], [0.7]], dtype=dtype)
    tau = np.array([0.95, 0.9], dtype=dtype)[:, None]
    L_path = np.array(2.0, dtype=dtype)

    L = reflectance_to_radiance_enmap(R, wl_nm, E0, cos_sun=cos_sun, tau=tau, L_path=L_path)
    assert L.dtype == np.result_type(dtype, np.float32)
    assert L.shape == (2, bands)

    R_back = radiance_to_reflectance_enmap(L, wl_nm, E0, cos_sun=cos_sun, tau=tau, L_path=L_path)
    assert R_back.dtype == np.result_type(dtype, np.float32)
    R_expected = np.broadcast_to(R, R_back.shape)
    np.testing.assert_allclose(R_back, R_expected, rtol=1e-4, atol=1e-4)
