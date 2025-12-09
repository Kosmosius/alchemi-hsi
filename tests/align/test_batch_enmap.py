import numpy as np

from alchemi.align.batch_builders import NoiseConfig, PairBatch, build_enmap_pairs
from alchemi.srf.batch_convolve import batch_convolve_lab_to_sensor

from alchemi.srf.enmap import build_enmap_sensor_srf
from alchemi.srf.registry import get_srf, register_sensor_srf
from alchemi.srf.resample import resample_values_with_srf


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


def _lab_batch() -> list[tuple[np.ndarray, np.ndarray]]:
    nm, spectra = _synthetic_lab()
    return [(nm, spectra[idx]) for idx in range(spectra.shape[0])]


def test_enmap_pairing_shapes_and_sampling(tmp_path) -> None:
    pairs = build_enmap_pairs(_lab_batch(), cache_dir=tmp_path)

    assert all(isinstance(pair, PairBatch) for pair in pairs)
    assert len(pairs) == 3

    band_grid = pairs[0].sensor_wavelengths_nm
    assert band_grid.shape == (226,)
    assert np.all(np.diff(band_grid) > 0)
    assert pairs[0].sensor_id == "enmap"

    vnir = band_grid[:95]
    swir = band_grid[95:]
    assert vnir.size == 95
    assert swir.size == 131
    assert np.isclose(np.mean(np.diff(vnir)), 6.16, atol=0.15)
    assert np.isclose(np.mean(np.diff(swir)), 11.15, atol=0.2)


def test_enmap_convolution_matches_reference(tmp_path) -> None:
    batch = _lab_batch()
    pairs = build_enmap_pairs(
        batch,
        cache_dir=tmp_path,
        noise_level_rel_vnir=0.0,
        noise_level_rel_swir=0.0,
    )

    sensor_srf = get_srf("enmap")
    if sensor_srf is None:
        sensor_srf = build_enmap_sensor_srf(cache_dir=tmp_path)
        register_sensor_srf(sensor_srf)
    expected, _ = resample_values_with_srf(np.stack([b[1] for b in batch]), batch[0][0], sensor_srf)

    stacked = np.stack([pair.sensor_values for pair in pairs], axis=0)
    np.testing.assert_allclose(stacked, expected, atol=5e-6)


def test_enmap_noise_levels_split(tmp_path) -> None:
    batch = _lab_batch()
    base_pairs = build_enmap_pairs(
        batch,
        cache_dir=tmp_path,
        noise_level_rel_vnir=0.0,
        noise_level_rel_swir=0.0,
    )
    noisy_pairs = build_enmap_pairs(
        batch,
        cache_dir=tmp_path,
        noise_level_rel_vnir=0.03,
        noise_level_rel_swir=0.06,
        noise_cfg=NoiseConfig(seed=123),
    )

    sensor_srf = get_srf("enmap") or build_enmap_sensor_srf(cache_dir=tmp_path)
    expected, _ = resample_values_with_srf(np.stack([b[1] for b in batch]), batch[0][0], sensor_srf)
    rel_levels = np.where(sensor_srf.band_centers_nm <= 999.0, 0.03, 0.06)
    sigma = np.abs(expected) * rel_levels.reshape(1, -1)

    rng = np.random.default_rng(123)
    expected_noise = rng.normal(loc=0.0, scale=1.0, size=expected.shape) * sigma

    base = np.stack([pair.sensor_values for pair in base_pairs], axis=0)
    noisy = np.stack([pair.sensor_values for pair in noisy_pairs], axis=0)
    np.testing.assert_allclose(noisy, base + expected_noise)

    diff = noisy - base
    vnir_rms = np.sqrt(np.mean(diff[:, :95] ** 2))
    swir_rms = np.sqrt(np.mean(diff[:, 95:] ** 2))
    assert swir_rms > vnir_rms
