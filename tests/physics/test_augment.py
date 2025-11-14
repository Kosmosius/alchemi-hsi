from __future__ import annotations

import numpy as np
import pytest

from alchemi.physics import augment_radiance, random_swirlike_atmosphere


def _emit_like_radiance(R: np.ndarray, E0: np.ndarray, cos_sun: float, tau: np.ndarray, L_path: float) -> np.ndarray:
    return tau * (E0 * cos_sun / np.pi) * R + L_path


def test_random_swirlike_atmosphere_properties():
    wl_nm = np.linspace(1000.0, 2500.0, 200)
    rng = np.random.default_rng(123)

    tau_vec, L_path = random_swirlike_atmosphere(wl_nm, rng)

    assert tau_vec.shape == wl_nm.shape
    assert np.all(tau_vec > 0.0)
    assert np.all(tau_vec <= 1.0)
    assert 0.0 <= L_path <= 5.0

    diffs = np.abs(np.diff(tau_vec))
    assert np.quantile(diffs, 0.95) < 0.1

    water_mask = np.zeros_like(wl_nm, dtype=bool)
    water_ranges = ((1350.0, 1450.0), (1850.0, 1950.0))
    for left_nm, right_nm in water_ranges:
        water_mask |= (wl_nm >= left_nm) & (wl_nm <= right_nm)
    diff_with = tau_vec[~water_mask].mean() - tau_vec[water_mask].mean()
    assert diff_with > 0.02

    rng = np.random.default_rng(123)
    tau_no_bands, _ = random_swirlike_atmosphere(wl_nm, rng, disable_water_bands=True)
    diff_no = tau_no_bands[~water_mask].mean() - tau_no_bands[water_mask].mean()
    assert diff_no < diff_with
    assert abs(diff_no) < 0.05


@pytest.mark.parametrize("shape", [(), (4,), (2, 3), (2, 3, 5)])
def test_augment_radiance_shapes(shape):
    wl_nm = np.linspace(1000.0, 2500.0, 50)
    rng = np.random.default_rng(0)

    L = rng.uniform(0.0, 100.0, size=shape + (wl_nm.size,))
    L_aug = augment_radiance(L, wl_nm, rng)

    assert L_aug.shape == L.shape
    assert np.all(L_aug >= 0.0)

    L_zero = augment_radiance(L, wl_nm, rng, strength=0.0)
    assert np.allclose(L_zero, L)

    L_full = augment_radiance(L, wl_nm, rng, strength=1.0)
    assert not np.allclose(L_full, L)


def test_augment_radiance_emit_sanity():
    rng = np.random.default_rng(42)
    wl_nm = np.linspace(1000.0, 2500.0, 150)
    cos_sun = 0.85
    L_path0 = 1.5

    R = rng.uniform(0.05, 0.7, size=(6, wl_nm.size))
    E0 = rng.uniform(100.0, 200.0, size=wl_nm.size)
    tau0 = rng.uniform(0.8, 1.0, size=wl_nm.size)

    L0 = _emit_like_radiance(R, E0, cos_sun, tau0, L_path0)
    L_aug = augment_radiance(L0, wl_nm, rng, strength=0.25)

    assert np.all(L_aug >= 0.0)
    assert np.all(L_aug < 1000.0)

    for i in range(L0.shape[0]):
        corr = np.corrcoef(L0[i], L_aug[i])[0, 1]
        assert corr > 0.98
