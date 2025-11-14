from __future__ import annotations

import numpy as np

from alchemi.physics.swir_emit import (
    band_depth,
    continuum_removed,
    radiance_to_reflectance_emit,
    reflectance_to_radiance_emit,
)


def test_round_trip_identity():
    wl = np.linspace(1000.0, 2500.0, 256)
    E0 = np.linspace(1500.0, 2000.0, wl.size)
    R_true = 0.1 + 0.35 * np.sin(wl / 200.0) ** 2

    L = reflectance_to_radiance_emit(R_true, wl, E0, cos_sun=1.0)
    R_est = radiance_to_reflectance_emit(L, wl, E0, cos_sun=1.0)

    assert np.max(np.abs(R_est - R_true)) < 1e-6


def test_path_radiance_sensitivity():
    wl = np.linspace(1100.0, 2400.0, 128)
    E0 = np.linspace(1400.0, 1900.0, wl.size)
    R = 0.2 + 0.05 * np.cos(wl / 180.0)
    tau = 0.85
    cos_sun = 0.9

    L0 = reflectance_to_radiance_emit(R, wl, E0, cos_sun, tau=tau, L_path=0.0)
    L1 = reflectance_to_radiance_emit(R, wl, E0, cos_sun, tau=tau, L_path=1.0)
    L5 = reflectance_to_radiance_emit(R, wl, E0, cos_sun, tau=tau, L_path=5.0)

    assert np.allclose(L1 - 1.0, L0)
    assert np.allclose(L5 - 5.0, L0)


def test_transmittance_inversion_sensitivity():
    wl = np.linspace(1200.0, 2400.0, 64)
    E0 = np.linspace(1300.0, 2000.0, wl.size)
    cos_sun = 0.8
    base_reflectance = 0.25 + 0.1 * np.sin(wl / 150.0)
    L = reflectance_to_radiance_emit(base_reflectance, wl, E0, cos_sun=cos_sun, tau=0.9)

    R_high_tau = radiance_to_reflectance_emit(L, wl, E0, cos_sun=cos_sun, tau=0.9)
    R_low_tau = radiance_to_reflectance_emit(L, wl, E0, cos_sun=cos_sun, tau=0.6)

    assert np.all(R_low_tau >= R_high_tau - 1e-9)
    assert np.any(R_low_tau > R_high_tau)


def test_band_depth_sanity():
    wl = np.linspace(2000.0, 2400.0, 201)
    sigma = 25.0
    amplitude = 0.08
    base = 0.45 + 0.02 * np.sin(wl / 80.0)
    dip = amplitude * np.exp(-0.5 * ((wl - 2200.0) / sigma) ** 2)
    spectrum = base - dip

    depth = band_depth(spectrum, wl, left_nm=2100.0, center_nm=2200.0, right_nm=2300.0)

    left_val = np.interp(2100.0, wl, spectrum)
    right_val = np.interp(2300.0, wl, spectrum)
    slope = (right_val - left_val) / (2300.0 - 2100.0)
    continuum_center = left_val + slope * (2200.0 - 2100.0)
    center_val = np.interp(2200.0, wl, spectrum)
    expected_depth = 1.0 - center_val / continuum_center

    assert np.isclose(float(depth), expected_depth, atol=0.02)

    deeper_spectrum = base - 1.5 * dip
    deeper_depth = band_depth(
        deeper_spectrum, wl, left_nm=2100.0, center_nm=2200.0, right_nm=2300.0
    )
    assert deeper_depth > depth


def test_dtype_safety():
    wl = np.linspace(1000.0, 2500.0, 128)
    E0 = np.linspace(1500.0, 2100.0, wl.size)
    R = 0.3 + 0.2 * np.sin(wl / 220.0)

    L64 = reflectance_to_radiance_emit(R, wl, E0, cos_sun=0.75, tau=0.95, L_path=0.5)
    L32 = reflectance_to_radiance_emit(
        R.astype(np.float32),
        wl.astype(np.float32),
        E0.astype(np.float32),
        cos_sun=np.float32(0.75),
        tau=np.float32(0.95),
        L_path=np.float32(0.5),
    )
    L16 = reflectance_to_radiance_emit(
        R.astype(np.float16),
        wl.astype(np.float16),
        E0.astype(np.float16),
        cos_sun=np.float16(0.75),
        tau=np.float16(0.95),
        L_path=np.float16(0.5),
    )

    assert np.all(np.isfinite(L32))
    assert np.all(np.isfinite(L16))
    assert np.allclose(L32, L64, rtol=1e-4, atol=1e-6)
    assert np.allclose(L16, L64, rtol=2e-3, atol=1e-4)

    R_est64 = radiance_to_reflectance_emit(L64, wl, E0, cos_sun=0.75, tau=0.95, L_path=0.5)
    R_est32 = radiance_to_reflectance_emit(
        L32.astype(np.float32),
        wl.astype(np.float32),
        E0.astype(np.float32),
        cos_sun=np.float32(0.75),
        tau=np.float32(0.95),
        L_path=np.float32(0.5),
    )
    R_est16 = radiance_to_reflectance_emit(
        L16.astype(np.float16),
        wl.astype(np.float16),
        E0.astype(np.float16),
        cos_sun=np.float16(0.75),
        tau=np.float16(0.95),
        L_path=np.float16(0.5),
    )

    assert np.all(np.isfinite(R_est32))
    assert np.all(np.isfinite(R_est16))
    assert np.allclose(R_est32, R_est64, rtol=1e-4, atol=1e-6)
    assert np.allclose(R_est16, R_est64, rtol=3e-3, atol=2e-4)

    removed = continuum_removed(R, wl)
    assert removed.shape == R.shape
