import numpy as np

from alchemi.align.batch_builders import NoiseConfig, build_avirisng_pairs
from alchemi.srf.avirisng import build_avirisng_sensor_srf
from alchemi.srf.registry import get_srf, register_sensor_srf
from alchemi.srf.resample import resample_values_with_srf


def _lab_batch(num: int = 256) -> list[tuple[np.ndarray, np.ndarray]]:
    nm = np.linspace(380.0, 2500.0, num, dtype=np.float64)
    spectra = np.stack(
        [
            0.2 + 0.05 * np.sin(nm / 75.0),
            0.3 + 0.02 * np.cos(nm / 50.0),
        ],
        axis=0,
    )
    return [(nm, spec) for spec in spectra]


def test_aviris_pairs_projection_matches(tmp_path) -> None:
    batch = _lab_batch()
    pairs = build_avirisng_pairs(batch, cache_dir=tmp_path)

    sensor_srf = get_srf("avirisng")
    if sensor_srf is None:
        sensor_srf = build_avirisng_sensor_srf(cache_dir=tmp_path)
        register_sensor_srf(sensor_srf)
    expected, _ = resample_values_with_srf(np.stack([b[1] for b in batch]), batch[0][0], sensor_srf)

    stacked = np.stack([pair.sensor_values for pair in pairs], axis=0)
    np.testing.assert_allclose(stacked, expected, atol=1e-6)
    assert pairs[0].sensor_mask is not None
    assert pairs[0].sensor_mask.shape == pairs[0].sensor_wavelengths_nm.shape


def test_aviris_noise_vector_application(tmp_path) -> None:
    batch = _lab_batch()
    base_pairs = build_avirisng_pairs(batch, cache_dir=tmp_path)

    srf = get_srf("avirisng") or build_avirisng_sensor_srf(cache_dir=tmp_path)
    band_noise = np.linspace(0.0, 0.05, srf.band_count, dtype=np.float64)
    noisy_pairs = build_avirisng_pairs(
        batch,
        cache_dir=tmp_path,
        noise=band_noise,
        noise_cfg=NoiseConfig(seed=7),
    )

    sensor_srf = get_srf("avirisng") or build_avirisng_sensor_srf(cache_dir=tmp_path)
    expected, _ = resample_values_with_srf(np.stack([b[1] for b in batch]), batch[0][0], sensor_srf)
    sigma = np.abs(expected) * band_noise.reshape(1, -1)
    rng = np.random.default_rng(7)
    expected_noise = rng.normal(loc=0.0, scale=1.0, size=expected.shape) * sigma

    base = np.stack([pair.sensor_values for pair in base_pairs], axis=0)
    noisy = np.stack([pair.sensor_values for pair in noisy_pairs], axis=0)
    np.testing.assert_allclose(noisy, base + expected_noise)
