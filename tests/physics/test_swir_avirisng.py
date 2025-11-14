from __future__ import annotations

import numpy as np
import numpy.testing as npt

from alchemi.physics.swir_avirisng import (
    avirisng_bad_band_mask,
    radiance_to_reflectance_avirisng,
    reflectance_to_radiance_avirisng,
)


def test_round_trip_identity_valid_bands() -> None:
    rng = np.random.default_rng(42)
    wl_nm = np.linspace(380.0, 2510.0, 425)
    mask = avirisng_bad_band_mask(wl_nm)

    R_true = np.full_like(wl_nm, np.nan, dtype=float)
    R_true[mask] = rng.uniform(0.1, 0.8, mask.sum())

    E0 = np.linspace(1500.0, 1800.0, wl_nm.size)

    L = reflectance_to_radiance_avirisng(
        R_true, wl_nm, E0, cos_sun=1.0, tau=1.0, L_path=0.0, band_mask=mask
    )
    R_est = radiance_to_reflectance_avirisng(
        L, wl_nm, E0, cos_sun=1.0, tau=1.0, L_path=0.0, band_mask=mask
    )

    npt.assert_allclose(R_est[mask], R_true[mask], atol=1e-6)
    assert np.all(np.isnan(R_est[~mask]))
    assert np.all(np.isnan(L[~mask]))


def test_bad_band_mask_ranges() -> None:
    wl_nm = np.array([1330, 1340, 1400, 1440, 1500, 1799, 1800, 1925, 1950, 1960])
    mask = avirisng_bad_band_mask(wl_nm)

    expected = np.array([True, False, False, False, True, True, False, False, False, True])
    npt.assert_array_equal(mask, expected)


def test_tau_and_lpath_sensitivity() -> None:
    wl_nm = np.linspace(1000.0, 2400.0, 20)
    mask = avirisng_bad_band_mask(wl_nm)

    R = np.linspace(0.2, 0.6, wl_nm.size)
    E0 = np.linspace(1200.0, 1900.0, wl_nm.size)
    cos_sun = 0.85
    tau = np.linspace(0.7, 1.2, wl_nm.size)
    L_path = np.linspace(0.01, 0.05, wl_nm.size)

    L = reflectance_to_radiance_avirisng(
        R, wl_nm, E0, cos_sun=cos_sun, tau=tau, L_path=L_path, band_mask=mask
    )

    expected_L = tau * (E0 * cos_sun / np.pi) * R + L_path
    npt.assert_allclose(L[mask], expected_L[mask], rtol=1e-12, atol=0)
    assert np.all(np.isnan(L[~mask]))

    R_recovered = radiance_to_reflectance_avirisng(
        L, wl_nm, E0, cos_sun=cos_sun, tau=tau, L_path=L_path, band_mask=mask
    )
    npt.assert_allclose(R_recovered[mask], R[mask], atol=1e-6)


def test_dtype_and_broadcast_behaviour() -> None:
    wl_nm = np.array([1000.0, 1400.0, 1500.0, 1800.0, 2100.0], dtype=np.float32)
    mask = avirisng_bad_band_mask(wl_nm)

    R = np.linspace(0.1, 0.5, wl_nm.size, dtype=np.float32)
    E0 = np.full(wl_nm.shape, 1500.0, dtype=np.float32)
    cos_sun = np.float32(0.9)
    tau = np.float32(0.95)
    L_path = np.float32(0.02)

    L = reflectance_to_radiance_avirisng(
        R, wl_nm, E0, cos_sun=cos_sun, tau=tau, L_path=L_path, band_mask=mask
    )
    assert L.dtype == np.float32

    R_back = radiance_to_reflectance_avirisng(
        L, wl_nm, E0, cos_sun=cos_sun, tau=tau, L_path=L_path, band_mask=mask
    )
    assert R_back.dtype == np.float32

    npt.assert_allclose(R_back[mask], R[mask], atol=1e-7)
    assert np.all(np.isnan(L[~mask]))
    assert np.all(np.isnan(R_back[~mask]))
