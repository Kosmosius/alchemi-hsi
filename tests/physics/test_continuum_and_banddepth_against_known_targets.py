import numpy as np

from alchemi.physics.continuum import compute_band_depth, compute_convex_hull_continuum
from alchemi.types import QuantityKind, Spectrum, WavelengthGrid


def test_band_depth_matches_synthetic_feature():
    wavelengths = WavelengthGrid(np.linspace(1000.0, 1010.0, 6))
    reflectance = np.array([1.0, 0.9, 0.8, 0.9, 1.0, 1.0])
    spectrum = Spectrum.from_reflectance(wavelengths, reflectance)
    continuum = compute_convex_hull_continuum(spectrum)
    assert continuum.kind == QuantityKind.REFLECTANCE
    depth = compute_band_depth(spectrum, lambda_center_nm=1004.0, lambda_left_nm=1000.0, lambda_right_nm=1010.0)
    assert depth > 0
    assert depth < 0.3


def test_convex_hull_returns_upper_envelope():
    wavelengths = WavelengthGrid(np.array([500.0, 600.0, 700.0]))
    reflectance = np.array([0.8, 0.5, 0.9])
    spectrum = Spectrum.from_reflectance(wavelengths, reflectance)
    continuum = compute_convex_hull_continuum(spectrum)
    assert np.all(continuum.values >= spectrum.values)
