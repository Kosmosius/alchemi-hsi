import numpy as np

from alchemi.align.batch_builders import NoiseConfig, SRFJitterConfig, build_emit_pairs


def _lab_grid(num: int = 1024) -> np.ndarray:
    return np.linspace(350.0, 2550.0, num, dtype=np.float64)


def test_emit_pairs_match_without_jitter() -> None:
    grid = _lab_grid()
    lab = np.exp(-0.5 * ((grid - 1500.0) / 50.0) ** 2)
    batch = [(grid, lab)]

    base_cfg = NoiseConfig(noise_level_rel=0.0, seed=0)
    base_pairs = build_emit_pairs(batch, noise_cfg=base_cfg)

    jitter_cfg = SRFJitterConfig(enabled=False, seed=1)
    jitter_pairs = build_emit_pairs(batch, noise_cfg=base_cfg, srf_jitter_cfg=jitter_cfg)

    np.testing.assert_allclose(base_pairs[0].sensor_values, jitter_pairs[0].sensor_values)
    np.testing.assert_allclose(
        base_pairs[0].sensor_wavelengths_nm, jitter_pairs[0].sensor_wavelengths_nm
    )


def test_emit_pairs_use_jittered_srf_when_enabled() -> None:
    grid = _lab_grid()
    lab = np.cos(grid / 200.0) ** 2
    batch = [(grid, lab)]

    cfg = SRFJitterConfig(enabled=True, center_shift_std_nm=2.0, seed=3)

    jittered_pairs = build_emit_pairs(batch, noise_cfg=NoiseConfig(), srf_jitter_cfg=cfg)
    base_pairs = build_emit_pairs(batch, noise_cfg=NoiseConfig())

    assert not np.allclose(
        jittered_pairs[0].sensor_wavelengths_nm, base_pairs[0].sensor_wavelengths_nm
    )
    assert not np.allclose(jittered_pairs[0].sensor_values, base_pairs[0].sensor_values)
