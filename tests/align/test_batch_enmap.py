from __future__ import annotations

import numpy as np

from alchemi.align import build_enmap_pairs
from alchemi.srf import batch_convolve_lab_to_sensor, enmap_srf_matrix


def _synthetic_lab() -> tuple[np.ndarray, np.ndarray]:
    nm = np.linspace(400.0, 2500.0, 4201, dtype=np.float64)
    base = np.sin(nm / 250.0) * 0.1 + 0.3
    slope = (nm - nm.min()) / (nm.max() - nm.min())
    spectra = np.stack(
        [
            base,
            0.25 + 0.05 * np.cos(nm / 180.0),
            0.2 + 0.1 * slope,
        ],
        axis=0,
    )
    return nm, spectra


def test_enmap_pairing_shapes_and_sampling(tmp_path):
    lab_nm, lab_reflectance = _synthetic_lab()
    wl, lab_conv, sensor, mask = build_enmap_pairs(
        lab_nm, lab_reflectance, cache_dir=tmp_path, rng=0
    )

    assert wl.shape == (226,)
    assert np.all(np.diff(wl) > 0)
    assert lab_conv.shape == (lab_reflectance.shape[0], wl.size)
    assert sensor.shape == lab_conv.shape
    assert mask.shape == (wl.size,)
    assert mask.dtype == bool and mask.all()

    vnir_centers = wl[:95]
    swir_centers = wl[95:]
    assert vnir_centers.size == 95
    assert swir_centers.size == 131
    assert np.isclose(np.mean(np.diff(vnir_centers)), 6.16, atol=0.15)
    assert np.isclose(np.mean(np.diff(swir_centers)), 11.15, atol=0.2)


def test_enmap_convolution_matches_reference(tmp_path):
    lab_nm, lab_reflectance = _synthetic_lab()
    _wl, lab_conv, sensor, _ = build_enmap_pairs(
        lab_nm,
        lab_reflectance,
        cache_dir=tmp_path,
        rng=0,
        noise_level_rel_vnir=0.0,
        noise_level_rel_swir=0.0,
    )
    srf = enmap_srf_matrix(cache_dir=tmp_path)
    expected = batch_convolve_lab_to_sensor(lab_nm, lab_reflectance, srf)
    assert np.allclose(lab_conv, expected, atol=5e-6)
    assert np.allclose(sensor, lab_conv)


def test_enmap_noise_levels_split(tmp_path):
    lab_nm, lab_reflectance = _synthetic_lab()
    wl, lab_conv, sensor, _ = build_enmap_pairs(
        lab_nm,
        lab_reflectance,
        cache_dir=tmp_path,
        noise_level_rel_vnir=0.03,
        noise_level_rel_swir=0.06,
        rng=123,
    )

    generator = np.random.default_rng(123)
    rel_levels = np.where(wl <= 999.0, 0.03, 0.06)
    sigma = rel_levels[None, :] * np.maximum(np.abs(lab_conv), 1e-8)
    expected_noise = generator.normal(loc=0.0, scale=sigma, size=lab_conv.shape)
    assert np.allclose(sensor, lab_conv + expected_noise)

    diff = sensor - lab_conv
    vnir_rms = np.sqrt(np.mean(diff[:, :95] ** 2))
    swir_rms = np.sqrt(np.mean(diff[:, 95:] ** 2))
    assert swir_rms > vnir_rms
