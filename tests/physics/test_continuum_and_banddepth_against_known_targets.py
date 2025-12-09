"""Regression tests for continuum removal and band-depth utilities.

Golden spectra and target values come from ``tests/data/golden_banddepth.json``.
They were generated offline with ``alchemi.physics.continuum`` using simple
synthetic absorptions (see the JSON description for provenance). These tests act
as a contract: small numerical drift is tolerated (rtol=1e-6, atol=1e-9) but
shape/metric changes should be intentional and accompanied by fixture updates.
The fixture covers moderate, shallow, and partially overlapping absorption bands
so that edge cases like near-zero depths and neighbouring features remain
stable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from alchemi.physics.continuum import (
    build_continuum,
    compute_band_metrics,
    compute_convex_hull_continuum,
    compute_band_depth,
    continuum_remove,
)
from alchemi.types import Spectrum, WavelengthGrid


pytestmark = pytest.mark.physics_and_metadata


def _load_golden():
    fixture_path = Path(__file__).resolve().parents[1] / "data" / "golden_banddepth.json"
    with fixture_path.open("r", encoding="utf-8") as f:
        content = json.load(f)
    return {entry["id"]: entry for entry in content["spectra"]}


golden_spectra = _load_golden()


@pytest.mark.parametrize("spectrum_id", list(golden_spectra))
def test_continuum_and_removed_match_golden(spectrum_id: str):
    entry = golden_spectra[spectrum_id]
    wavelengths = WavelengthGrid(np.array(entry["wavelength_nm"], dtype=np.float64))
    reflectance = np.array(entry["reflectance"], dtype=np.float64)
    spectrum = Spectrum.from_reflectance(wavelengths, reflectance)

    continuum = build_continuum(spectrum)
    removed = continuum_remove(spectrum)

    assert continuum.shape == reflectance.shape
    assert removed.values.shape == reflectance.shape
    np.testing.assert_allclose(continuum, entry["continuum"], rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(removed.values, entry["continuum_removed"], rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize(
    "spectrum_id, band_idx",
    [
        ("synthetic_moderate_band", 0),
        ("synthetic_shallow_band", 0),
        ("synthetic_overlapping_bands", 0),
        ("synthetic_overlapping_bands", 1),
    ],
)
def test_band_metrics_match_golden(spectrum_id: str, band_idx: int):
    entry = golden_spectra[spectrum_id]
    band = entry["bands"][band_idx]
    wavelengths = WavelengthGrid(np.array(entry["wavelength_nm"], dtype=np.float64))
    reflectance = np.array(entry["reflectance"], dtype=np.float64)
    spectrum = Spectrum.from_reflectance(wavelengths, reflectance)

    metrics = compute_band_metrics(
        spectrum,
        lambda_left_nm=band["lambda_left_nm"],
        lambda_center_nm=band["lambda_center_nm"],
        lambda_right_nm=band["lambda_right_nm"],
    )

    np.testing.assert_allclose(metrics.depth, band["expected_depth"], rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(metrics.area, band["expected_area"], rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(metrics.asymmetry, band["expected_asymmetry"], rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize(
    "spectrum_id, band_idx", [("synthetic_moderate_band", 0), ("synthetic_shallow_band", 0)]
)
def test_backward_compatible_depth_helper(spectrum_id: str, band_idx: int):
    entry = golden_spectra[spectrum_id]
    band = entry["bands"][band_idx]
    wavelengths = WavelengthGrid(np.array(entry["wavelength_nm"], dtype=np.float64))
    reflectance = np.array(entry["reflectance"], dtype=np.float64)
    spectrum = Spectrum.from_reflectance(wavelengths, reflectance)

    depth = compute_band_depth(
        spectrum,
        lambda_center_nm=band["lambda_center_nm"],
        lambda_left_nm=band["lambda_left_nm"],
        lambda_right_nm=band["lambda_right_nm"],
    )

    np.testing.assert_allclose(depth, band["expected_depth"], rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize("spectrum_id", ["synthetic_shallow_band", "synthetic_overlapping_bands"])
def test_convex_hull_continuum_is_upper_envelope(spectrum_id: str):
    entry = golden_spectra[spectrum_id]
    wavelengths = WavelengthGrid(np.array(entry["wavelength_nm"], dtype=np.float64))
    reflectance = np.array(entry["reflectance"], dtype=np.float64)
    spectrum = Spectrum.from_reflectance(wavelengths, reflectance)

    continuum = compute_convex_hull_continuum(spectrum)
    assert np.all(continuum.values >= spectrum.values)

    # Shallow/overlapping cases should not introduce NaNs in downstream metrics
    for band in entry["bands"]:
        metrics = compute_band_metrics(
            spectrum,
            lambda_left_nm=band["lambda_left_nm"],
            lambda_center_nm=band["lambda_center_nm"],
            lambda_right_nm=band["lambda_right_nm"],
            continuum=continuum.values,
        )
        assert not np.isnan(metrics.depth)
        assert not np.isnan(metrics.area)
        assert not np.isnan(metrics.asymmetry)
