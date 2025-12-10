"""Regression tests for EMIT-style band depths using canonical targets.

The JSON fixture ``tests/data/emit_banddepth_targets.json`` stores synthetic
spectra and pre-computed band depths/areas using anchor-based continua that
mirror EMIT L2B mineral definitions. The goal is to ensure continuum removal
and band-depth calculations remain numerically stable for realistic EMIT-like
windows (Al-OH ~2200 nm and carbonate ~2330 nm).

TODO: incorporate full EMIT L2B preprocessing (e.g., cloud/ice masking and
spectral response handling) once lightweight fixtures are available. The
current tests focus solely on continuum removal/band metrics and deliberately
omit product-level masking nuances.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from alchemi.physics.continuum import compute_band_metrics
from alchemi.types import BandDefinition, Spectrum, WavelengthGrid


pytestmark = pytest.mark.physics_and_metadata


def _load_emit_fixture() -> dict:
    fixture_path = Path(__file__).resolve().parents[1] / "data" / "emit_banddepth_targets.json"
    with fixture_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _band_definitions(config: dict) -> dict[str, BandDefinition]:
    definitions = {}
    for band in config.get("bands", []):
        definitions[band["name"]] = BandDefinition(
            lambda_center_nm=float(band["lambda_center_nm"]),
            lambda_left_nm=float(band["lambda_left_nm"]),
            lambda_right_nm=float(band["lambda_right_nm"]),
            name=band.get("name"),
        )
    return definitions


_emit_fixture = _load_emit_fixture()
_emit_band_defs = _band_definitions(_emit_fixture)


@pytest.mark.parametrize("target", _emit_fixture["targets"])
def test_emit_band_depths_match_fixture(target: dict) -> None:
    wavelengths = WavelengthGrid(np.asarray(target["wavelength_nm"], dtype=np.float64))
    reflectance = np.asarray(target["reflectance"], dtype=np.float64)
    spectrum = Spectrum.from_surface_reflectance(wavelengths, reflectance)

    for band_name, expected in target["expected"].items():
        band = _emit_band_defs[band_name]
        metrics = compute_band_metrics(
            spectrum,
            band=band,
            method="anchors",
            anchors=[(band.lambda_left_nm, band.lambda_right_nm)],
        )
        np.testing.assert_allclose(metrics.depth, expected["depth"], atol=5e-6, rtol=0)
        np.testing.assert_allclose(metrics.area, expected["area"], atol=5e-6, rtol=0)


@pytest.mark.parametrize("band_name", [entry["name"] for entry in _emit_fixture["bands"]])
def test_emit_band_depths_are_consistent_across_targets(band_name: str) -> None:
    """Ensure continuum handling stays stable across diverse EMIT-like spectra."""

    band = _emit_band_defs[band_name]
    depths = []
    for target in _emit_fixture["targets"]:
        wavelengths = WavelengthGrid(np.asarray(target["wavelength_nm"], dtype=np.float64))
        reflectance = np.asarray(target["reflectance"], dtype=np.float64)
        spectrum = Spectrum.from_surface_reflectance(wavelengths, reflectance)
        metrics = compute_band_metrics(
            spectrum,
            band=band,
            method="anchors",
            anchors=[(band.lambda_left_nm, band.lambda_right_nm)],
        )
        depths.append(float(metrics.depth))

    # EMIT L2B definitions should produce non-negative depths with stable ordering
    assert all(depth >= 0 for depth in depths)
    assert depths == sorted(depths, reverse=True)
