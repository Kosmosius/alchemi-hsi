"""Tests for continuum removal and band-depth utilities."""

from __future__ import annotations

import numpy as np

from alchemi.physics.continuum import (
    BandMetrics,
    build_continuum,
    compute_band_metrics,
    continuum_remove,
)
from alchemi.types import BandDefinition, Spectrum, WavelengthGrid


def _synthetic_reflectance(wavelengths: np.ndarray) -> np.ndarray:
    base = 0.4 + 0.0002 * (wavelengths - wavelengths[0])
    absorption1 = 0.1 * np.exp(-0.5 * ((wavelengths - 1400) / 20) ** 2)
    absorption2 = 0.15 * np.exp(-0.5 * ((wavelengths - 1800) / 30) ** 2)
    return base - absorption1 - absorption2


def test_convex_hull_continuum_tracks_upper_envelope() -> None:
    wavelengths = np.linspace(1200, 2000, 200)
    reflectance = _synthetic_reflectance(wavelengths)
    spectrum = Spectrum.from_reflectance(WavelengthGrid(wavelengths), reflectance)

    continuum = build_continuum(spectrum, method="convex_hull")
    removed = continuum_remove(spectrum, method="convex_hull")

    # Outside absorption regions, continuum-removed spectrum should be ~1
    outside = (wavelengths < 1360) | (wavelengths > 1880)
    assert np.all((removed.values[outside] > 0.97) & (removed.values[outside] < 1.03))

    # Continuum should sit above or on top of the reflectance curve
    assert np.all(continuum >= reflectance - 1e-9)


def test_anchor_based_continuum_matches_linear_segments() -> None:
    wavelengths = np.linspace(1000, 2000, 150)
    baseline = 0.3 + 0.0001 * (wavelengths - wavelengths[0])
    reflectance = baseline.copy()
    reflectance[60:80] -= 0.05
    spectrum = Spectrum.from_reflectance(WavelengthGrid(wavelengths), reflectance)

    anchors = [(1000.0, 1300.0), (1700.0, 2000.0)]
    continuum = build_continuum(spectrum, method="anchors", anchors=anchors)

    # Continuum should follow the baseline implied by the anchors
    left_vals = np.interp([anchors[0][0], anchors[0][1]], wavelengths, baseline)
    right_vals = np.interp([anchors[1][0], anchors[1][1]], wavelengths, baseline)
    expected_left = np.interp(1200.0, [anchors[0][0], anchors[0][1]], left_vals)
    expected_right = np.interp(1850.0, [anchors[1][0], anchors[1][1]], right_vals)

    assert np.isclose(np.interp(1200.0, wavelengths, continuum), expected_left, atol=1e-6)
    assert np.isclose(np.interp(1850.0, wavelengths, continuum), expected_right, atol=1e-6)


def test_band_metrics_gaussian_feature() -> None:
    wavelengths = np.linspace(1300, 1700, 300)
    depth_true = 0.2
    sigma = 50.0
    absorption = depth_true * np.exp(-0.5 * ((wavelengths - 1500.0) / sigma) ** 2)
    reflectance = 1.0 - absorption
    spectrum = Spectrum.from_reflectance(WavelengthGrid(wavelengths), reflectance)

    metrics = compute_band_metrics(
        spectrum,
        band=BandDefinition(lambda_center_nm=1500.0, lambda_left_nm=1350.0, lambda_right_nm=1650.0),
    )

    expected_area = float(
        np.trapezoid(
            absorption[(wavelengths >= 1350) & (wavelengths <= 1650)],
            wavelengths[(wavelengths >= 1350) & (wavelengths <= 1650)],
        )
    )

    assert isinstance(metrics, BandMetrics)
    assert np.isclose(metrics.depth, depth_true, atol=2e-3)
    assert np.isclose(metrics.area, expected_area, rtol=1e-2)
    assert np.isclose(metrics.asymmetry, 1.0, atol=1e-2)

    # Skewed feature should shift asymmetry away from unity
    sigma_left, sigma_right = 30.0, 80.0
    skew_absorption = np.where(
        wavelengths <= 1500,
        depth_true * np.exp(-0.5 * ((wavelengths - 1500.0) / sigma_left) ** 2),
        depth_true * np.exp(-0.5 * ((wavelengths - 1500.0) / sigma_right) ** 2),
    )
    skew_spectrum = Spectrum.from_reflectance(WavelengthGrid(wavelengths), 1.0 - skew_absorption)
    skew_metrics = compute_band_metrics(
        skew_spectrum,
        band=BandDefinition(lambda_center_nm=1500.0, lambda_left_nm=1350.0, lambda_right_nm=1650.0),
    )

    assert skew_metrics.asymmetry < 0.9
