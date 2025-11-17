import numpy as np

from alchemi.align.batch_builders import NoiseConfig, build_emits_pairs
from alchemi.srf.registry import get_srf
from alchemi.srf.resample import project_to_sensor


def _lab_grid(num: int = 1024) -> np.ndarray:
    return np.linspace(350.0, 2550.0, num, dtype=np.float64)


def test_emits_pairs_band_geometry() -> None:
    grid = _lab_grid()
    srf_matrix, _ = get_srf("emit", wavelengths_nm=grid)
    lab_batch = [
        (grid, np.exp(-0.5 * ((grid - 600.0) / 35.0) ** 2)),
        (grid, np.exp(-0.5 * ((grid - 1200.0) / 80.0) ** 2)),
    ]

    pairs = build_emits_pairs(lab_batch)
    assert len(pairs) == len(lab_batch)
    expected_bands = srf_matrix.centers_nm.size
    for pair in pairs:
        np.testing.assert_allclose(pair.lab_wavelengths_nm, grid)
        assert pair.sensor_wavelengths_nm.shape[0] == expected_bands
        assert pair.sensor_values.shape[0] == expected_bands
        assert np.all(np.diff(pair.sensor_wavelengths_nm) > 0)


def test_emits_pairs_projection_matches_reference() -> None:
    grid = _lab_grid()
    lab = np.exp(-0.5 * ((grid - 1500.0) / 50.0) ** 2)
    pairs = build_emits_pairs([(grid, lab)], noise_cfg=NoiseConfig())
    (pair,) = pairs

    srf_matrix, _ = get_srf("emit", wavelengths_nm=grid)
    reference = project_to_sensor(grid, lab, srf_matrix.centers_nm, srf=srf_matrix)
    np.testing.assert_allclose(pair.sensor_values, reference, atol=1e-6)


def test_emits_pairs_noise_variance_control() -> None:
    grid = _lab_grid()
    lab = np.sin(grid / 150.0) ** 2 + 0.5
    batch = [(grid, lab) for _ in range(512)]

    base_pairs = build_emits_pairs(batch)
    base_values = np.stack([p.sensor_values for p in base_pairs], axis=0)

    noise_level = 0.05
    noise_cfg = NoiseConfig(noise_level_rel=noise_level, seed=123)
    noisy_pairs = build_emits_pairs(batch, noise_cfg=noise_cfg)
    noisy_values = np.stack([p.sensor_values for p in noisy_pairs], axis=0)

    delta = noisy_values - base_values
    sample_var = delta.var(axis=0, ddof=1)
    expected_var = (np.abs(base_values[0]) * noise_level) ** 2
    mask = expected_var > 1e-12
    assert mask.any()
    np.testing.assert_allclose(sample_var[mask], expected_var[mask], rtol=0.2, atol=1e-12)
