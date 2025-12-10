"""Tests for virtual sensor and SRF perturbation utilities."""

from __future__ import annotations

import numpy as np

from alchemi.physics.resampling import SyntheticSensorConfig, simulate_virtual_sensor


def test_virtual_sensor_shapes_and_reproducibility() -> None:
    wavelengths = np.linspace(400.0, 900.0, 501)
    lab_spectra = np.random.default_rng(7).random((3, wavelengths.size))

    cfg = SyntheticSensorConfig(
        (410.0, 880.0), n_bands=8, min_fwhm_nm=5.0, max_fwhm_nm=20.0, seed=5
    )
    centers, srf, band_spectra = simulate_virtual_sensor(wavelengths, lab_spectra, cfg)
    centers_again, srf_again, band_spectra_again = simulate_virtual_sensor(wavelengths, lab_spectra, cfg)

    assert centers.shape == (cfg.n_bands,)
    assert srf.matrix.shape == (cfg.n_bands, wavelengths.size)
    assert band_spectra.shape == (lab_spectra.shape[0], cfg.n_bands)
    assert np.all(centers >= cfg.spectral_range_nm[0])
    assert np.all(centers <= cfg.spectral_range_nm[1])

    np.testing.assert_allclose(centers, centers_again)
    np.testing.assert_allclose(srf.matrix, srf_again.matrix)
    np.testing.assert_allclose(band_spectra, band_spectra_again)


def test_virtual_sensor_perturbations_change_srf_and_output() -> None:
    wavelengths = np.linspace(500.0, 1200.0, 701)
    lab_spectra = np.sin(wavelengths / 100.0)[np.newaxis, :] + 1.5

    cfg = SyntheticSensorConfig(
        (520.0, 1100.0), n_bands=6, min_fwhm_nm=8.0, max_fwhm_nm=25.0, seed=11
    )
    base_centers, base_srf, base_bands = simulate_virtual_sensor(
        wavelengths, lab_spectra, cfg
    )
    pert_centers, pert_srf, pert_bands = simulate_virtual_sensor(
        wavelengths,
        lab_spectra,
        cfg,
        perturb_centers_nm=5.0,
        perturb_width_factor=(0.2, 0.5),
    )

    assert not np.allclose(base_srf.matrix, pert_srf.matrix)
    assert not np.allclose(base_bands, pert_bands)
    assert np.all(np.isfinite(pert_bands))
    assert np.all(pert_centers >= cfg.spectral_range_nm[0])
    assert np.all(pert_centers <= cfg.spectral_range_nm[1])


def test_virtual_sensor_preserves_flat_spectrum() -> None:
    wavelengths = np.linspace(600.0, 900.0, 301)
    lab_spectra = np.full((2, wavelengths.size), 7.5)

    cfg = SyntheticSensorConfig(
        (610.0, 890.0), n_bands=5, min_fwhm_nm=6.0, max_fwhm_nm=12.0, seed=21
    )
    _, _, band_spectra = simulate_virtual_sensor(
        wavelengths, lab_spectra, cfg, perturb_width_factor=0.1
    )

    expected = np.full((lab_spectra.shape[0], cfg.n_bands), 7.5)
    np.testing.assert_allclose(band_spectra, expected, atol=1e-6, rtol=1e-6)
