"""Unit tests for Planck-law radiance/temperature conversions."""

from __future__ import annotations

import numpy as np

from alchemi.physics import bt_K_to_radiance, radiance_to_bt_K


def _lw_wavelengths(num: int = 64) -> np.ndarray:
    return np.linspace(7_500.0, 12_000.0, num, dtype=np.float64)


def test_bt_to_radiance_round_trip_K():
    wl_nm = _lw_wavelengths()
    temps = np.linspace(240.0, 340.0, 13, dtype=np.float64)

    temp_grid = temps[:, None]
    wl_grid = wl_nm[None, :]

    radiance = bt_K_to_radiance(temp_grid, wl_grid)
    recovered = radiance_to_bt_K(radiance, wl_grid)

    max_err = np.max(np.abs(recovered - temp_grid))
    assert max_err < 0.1


def test_radiance_round_trip_relative_error():
    wl_nm = _lw_wavelengths()
    temps = np.linspace(230.0, 360.0, 17, dtype=np.float64)

    temp_grid = temps[:, None]
    wl_grid = wl_nm[None, :]

    radiance = bt_K_to_radiance(temp_grid, wl_grid)
    recovered_radiance = bt_K_to_radiance(
        radiance_to_bt_K(radiance, wl_grid),
        wl_grid,
    )

    with np.errstate(invalid="ignore"):
        rel_err = np.abs((recovered_radiance - radiance) / radiance)

    mask = radiance > 1e-20
    assert np.all(rel_err[mask] < 1e-6)


def test_radiance_monotonic_in_temperature():
    wl_nm = np.array([10_000.0], dtype=np.float64)
    temps = np.linspace(200.0, 350.0, 64, dtype=np.float64)

    radiance = bt_K_to_radiance(temps, wl_nm)
    radiance = np.squeeze(radiance)

    diffs = np.diff(radiance)
    assert np.all(diffs > 0.0)


def test_fp16_fp32_numerical_safety():
    wl = _lw_wavelengths()
    temps = np.linspace(260.0, 320.0, 7, dtype=np.float64)

    wl32 = wl.astype(np.float32)
    temps32 = temps.astype(np.float32)
    wl16 = wl.astype(np.float16)
    temps16 = temps.astype(np.float16)

    radiance64 = bt_K_to_radiance(temps[:, None], wl[None, :])
    bt64 = radiance_to_bt_K(radiance64, wl[None, :])

    radiance32 = bt_K_to_radiance(temps32[:, None], wl32[None, :])
    bt32 = radiance_to_bt_K(radiance32, wl32[None, :])

    radiance16 = bt_K_to_radiance(temps16[:, None], wl16[None, :])
    bt16 = radiance_to_bt_K(radiance16, wl16[None, :])

    assert np.all(np.isfinite(radiance32))
    assert np.all(np.isfinite(bt32))
    assert np.all(np.isfinite(radiance16))
    assert np.all(np.isfinite(bt16))

    np.testing.assert_allclose(
        radiance32.astype(np.float64),
        radiance64,
        rtol=5e-4,
        atol=5e-7,
    )
    np.testing.assert_allclose(
        bt32.astype(np.float64),
        bt64,
        rtol=5e-4,
        atol=5e-4,
    )

    np.testing.assert_allclose(
        radiance16.astype(np.float64),
        radiance64,
        rtol=5e-3,
        atol=5e-6,
    )
    np.testing.assert_allclose(
        bt16.astype(np.float64),
        bt64,
        rtol=5e-3,
        atol=5e-3,
    )
