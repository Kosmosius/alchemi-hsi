from __future__ import annotations

import numpy as np

from alchemi.physics.continuum import (
    compute_anchor_continuum,
    compute_convex_hull_continuum,
    continuum_remove,
    smooth_continuum,
)
from alchemi.types import Spectrum, WavelengthGrid


def _make_spectrum(wavelengths: np.ndarray, reflectance: np.ndarray) -> Spectrum:
    return Spectrum.from_surface_reflectance(WavelengthGrid(wavelengths), reflectance)


def test_convex_hull_matches_sloping_background() -> None:
    wavelengths = np.linspace(1000.0, 2000.0, num=200)
    background = 0.25 + 0.1 * (wavelengths - wavelengths[0]) / (wavelengths[-1] - wavelengths[0])
    absorption = 0.05 * np.exp(-0.5 * ((wavelengths - 1500.0) / 30.0) ** 2)
    reflectance = background - absorption

    spectrum = _make_spectrum(wavelengths, reflectance)
    continuum = compute_convex_hull_continuum(spectrum)

    assert np.all(continuum.values >= reflectance)
    np.testing.assert_allclose(continuum.values, background, atol=2e-3)


def test_anchor_continuum_respects_shoulders_and_indices() -> None:
    wavelengths = np.linspace(500.0, 700.0, num=21)
    left_shoulder = 0.35
    right_shoulder = 0.45
    true_continuum = np.interp(
        wavelengths, [wavelengths[0], wavelengths[-1]], [left_shoulder, right_shoulder]
    )
    absorption = 0.05 * np.exp(-0.5 * ((wavelengths - 600.0) / 8.0) ** 2)
    reflectance = true_continuum - absorption

    spectrum = _make_spectrum(wavelengths, reflectance)
    continuum = compute_anchor_continuum(spectrum, anchors=[0, wavelengths[-1]])

    np.testing.assert_allclose(continuum.values[0], left_shoulder)
    np.testing.assert_allclose(continuum.values[-1], right_shoulder)
    np.testing.assert_allclose(continuum.values, true_continuum, atol=1e-6)


def test_smoothing_reduces_high_frequency_noise() -> None:
    wavelengths = np.linspace(900.0, 1100.0, num=100)
    baseline = 0.3 + 0.05 * np.sin(2 * np.pi * (wavelengths - 900.0) / 400.0)
    noisy = baseline + 0.01 * np.sin(2 * np.pi * wavelengths / 10.0)
    continuum = _make_spectrum(wavelengths, noisy)

    smoothed = smooth_continuum(continuum, method="poly", order=3)

    residual_noise_before = np.std(noisy - baseline)
    residual_noise_after = np.std(smoothed.values - baseline)
    assert residual_noise_after < residual_noise_before
    np.testing.assert_allclose(smoothed.values, baseline, atol=5e-3)


def test_continuum_remove_accepts_precomputed_continuum() -> None:
    wavelengths = np.linspace(400.0, 800.0, num=51)
    continuum_vals = np.interp(wavelengths, [400.0, 800.0], [0.4, 0.5])
    absorption = 0.08 * np.exp(-0.5 * ((wavelengths - 600.0) / 15.0) ** 2)
    reflectance = continuum_vals - absorption

    spectrum = _make_spectrum(wavelengths, reflectance)
    continuum_spec = _make_spectrum(wavelengths, continuum_vals)
    removed = continuum_remove(spectrum, continuum=continuum_spec)

    np.testing.assert_allclose(removed.values.max(), 1.0, atol=1e-6)
    assert np.all(removed.values <= 1.0 + 1e-6)
