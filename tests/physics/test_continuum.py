from __future__ import annotations

import numpy as np
import pytest

from alchemi.physics.swir import band_depth, continuum_remove


def test_continuum_handles_edge_windows() -> None:
    wavelengths = np.linspace(400.0, 700.0, num=7)
    reflectance = np.linspace(1.0, 0.6, num=wavelengths.size)

    cont, removed = continuum_remove(wavelengths, reflectance, left_nm=400.0, right_nm=700.0)

    assert np.isfinite(cont).all()
    assert np.isfinite(removed).all()
    assert cont.shape == reflectance.shape


def test_band_depth_flat_spectrum_remains_near_zero() -> None:
    wavelengths = np.linspace(1000.0, 1300.0, num=16)
    reflectance = np.ones_like(wavelengths)

    depth = band_depth(
        wavelength_nm=wavelengths,
        reflectance=reflectance,
        center_nm=1120.0,
        left_nm=1040.0,
        right_nm=1250.0,
    )

    assert depth == pytest.approx(0.0, abs=1e-6)


def test_continuum_respects_nearby_boundaries() -> None:
    wavelengths = np.array([500.0, 505.0, 510.0, 515.0])
    reflectance = np.array([1.0, 0.8, 0.9, 1.0])

    cont, removed = continuum_remove(wavelengths, reflectance, left_nm=500.1, right_nm=514.9)

    assert np.isfinite(cont).all()
    assert np.isfinite(removed).all()
    assert removed.min() >= 0.0
