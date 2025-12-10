from __future__ import annotations

import numpy as np

from alchemi.physics.continuum import (
    BandMetrics,
    compute_band_area,
    compute_band_asymmetry,
    compute_band_depth,
    compute_band_metrics,
    compute_composite_depth_ratio,
    continuum_remove,
)
from alchemi.types import BandDefinition, Spectrum, WavelengthGrid


def _continuum_removed_spectrum(wavelengths: np.ndarray, reflectance: np.ndarray) -> Spectrum:
    spectrum = Spectrum.from_reflectance(WavelengthGrid(wavelengths), reflectance)
    continuum = np.ones_like(reflectance)
    return continuum_remove(spectrum, continuum=continuum)


def test_band_metrics_gaussian_feature_against_ground_truth() -> None:
    wavelengths = np.linspace(2000.0, 2400.0, 301)
    depth_true = 0.35
    sigma = 25.0
    absorption = depth_true * np.exp(-0.5 * ((wavelengths - 2200.0) / sigma) ** 2)
    reflectance = 1.0 - absorption
    continuum_removed = _continuum_removed_spectrum(wavelengths, reflectance)
    band = BandDefinition(lambda_center_nm=2200.0, lambda_left_nm=2140.0, lambda_right_nm=2260.0)

    metrics = compute_band_metrics(continuum_removed, band=band)
    expected_area = float(
        np.trapezoid(
            absorption[(wavelengths >= band.lambda_left_nm) & (wavelengths <= band.lambda_right_nm)],
            wavelengths[(wavelengths >= band.lambda_left_nm) & (wavelengths <= band.lambda_right_nm)],
        )
    )

    assert isinstance(metrics, BandMetrics)
    assert np.isclose(metrics.depth, depth_true, atol=1e-3)
    assert np.isclose(metrics.area, expected_area, rtol=1e-3)
    assert np.isclose(metrics.asymmetry, 1.0, atol=5e-3)

    # Standalone metric helpers should agree with the composite dataclass
    assert np.isclose(compute_band_depth(continuum_removed, band=band), metrics.depth)
    assert np.isclose(compute_band_area(continuum_removed, band=band), metrics.area)
    assert np.isclose(compute_band_asymmetry(continuum_removed, band=band), metrics.asymmetry)


def test_composite_depth_ratio_for_two_bands() -> None:
    wavelengths = np.linspace(1000.0, 1300.0, 121)
    depth_a, depth_b = 0.25, 0.1
    absorption_a = depth_a * np.exp(-0.5 * ((wavelengths - 1100.0) / 12.0) ** 2)
    absorption_b = depth_b * np.exp(-0.5 * ((wavelengths - 1220.0) / 15.0) ** 2)
    reflectance = 1.0 - (absorption_a + absorption_b)
    continuum_removed = _continuum_removed_spectrum(wavelengths, reflectance)

    band_a = BandDefinition(lambda_center_nm=1100.0, lambda_left_nm=1080.0, lambda_right_nm=1120.0, name="A")
    band_b = BandDefinition(lambda_center_nm=1220.0, lambda_left_nm=1200.0, lambda_right_nm=1240.0, name="B")

    metrics_a = compute_band_metrics(continuum_removed, band=band_a)
    metrics_b = compute_band_metrics(continuum_removed, band=band_b)

    ratios = compute_composite_depth_ratio({"A": metrics_a, "B": metrics_b}, {"A": 1.0}, {"B": 1.0})
    assert np.isclose(ratios, depth_a / depth_b, rtol=1e-2)

    # Weighted numerator without denominator should reduce to weighted sum
    weighted = compute_composite_depth_ratio({"A": metrics_a, "B": metrics_b}, {"A": 0.5, "B": 0.5})
    assert np.isclose(weighted, 0.5 * (depth_a + depth_b), rtol=1e-3)
