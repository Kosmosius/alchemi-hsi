import numpy as np

from alchemi.align.batch_builders import NoiseConfig, build_emit_pairs
from alchemi.srf.emit import build_emit_sensor_srf
from alchemi.srf.registry import get_srf, register_sensor_srf
from alchemi.srf.resample import resample_values_with_srf


def _lab_grid(num: int = 1024) -> np.ndarray:
    return np.linspace(350.0, 2550.0, num, dtype=np.float64)


def test_emits_pairs_band_geometry() -> None:
    grid = _lab_grid()
    sensor_srf = get_srf("emit")
    if sensor_srf is None:
        sensor_srf = build_emit_sensor_srf(wavelength_grid_nm=grid)
        register_sensor_srf(sensor_srf)
    lab_batch = [
        (grid, np.exp(-0.5 * ((grid - 600.0) / 35.0) ** 2)),
        (grid, np.exp(-0.5 * ((grid - 1200.0) / 80.0) ** 2)),
    ]

    pairs = build_emit_pairs(lab_batch)
    assert len(pairs) == len(lab_batch)
    expected_bands = sensor_srf.band_count
    for pair in pairs:
        np.testing.assert_allclose(pair.lab_wavelengths_nm, grid)
        assert pair.sensor_wavelengths_nm.shape[0] == expected_bands
        assert pair.sensor_values.shape[0] == expected_bands
        assert np.all(np.diff(pair.sensor_wavelengths_nm) > 0)
        assert pair.sensor_id == "emit"


def test_emits_pairs_projection_matches_reference() -> None:
    grid = _lab_grid()
    lab = np.exp(-0.5 * ((grid - 1500.0) / 50.0) ** 2)
    pairs = build_emit_pairs([(grid, lab)], noise_cfg=NoiseConfig())
    (pair,) = pairs

    sensor_srf = get_srf("emit") or build_emit_sensor_srf(wavelength_grid_nm=grid)
    expected, _ = resample_values_with_srf(lab, grid, sensor_srf)
    np.testing.assert_allclose(pair.sensor_values, expected, atol=1e-6)


def test_emits_pairs_noise_variance_control() -> None:
    grid = _lab_grid()
    lab = np.sin(grid / 150.0) ** 2 + 0.5
    batch = [(grid, lab) for _ in range(512)]

    base_pairs = build_emit_pairs(batch)
    base_values = np.stack([p.sensor_values for p in base_pairs], axis=0)

    noise_level = 0.05
    noise_cfg = NoiseConfig(noise_level_rel=noise_level, seed=123)
    noisy_pairs = build_emit_pairs(batch, noise_cfg=noise_cfg)
    noisy_values = np.stack([p.sensor_values for p in noisy_pairs], axis=0)

    delta = noisy_values - base_values
    sample_var = delta.var(axis=0, ddof=1)
    expected_var = (np.abs(base_values[0]) * noise_level) ** 2
    mask = expected_var > 1e-12
    assert mask.any()
    np.testing.assert_allclose(sample_var[mask], expected_var[mask], rtol=0.2, atol=1e-12)
