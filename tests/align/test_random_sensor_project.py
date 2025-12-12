import numpy as np
import pytest

from alchemi.align import RandomSensorProject
from alchemi.srf.synthetic import SyntheticSensorConfig


def _config() -> SyntheticSensorConfig:
    highres = np.linspace(400.0, 2500.0, 512)
    return SyntheticSensorConfig(
        highres_axis_nm=highres,
        n_bands=24,
        center_jitter_nm=6.0,
        fwhm_range_nm=(8.0, 20.0),
        shape="gaussian",
        seed=2024,
    )


@pytest.mark.parametrize("mode", ["per_sample", "per_batch", "fixed"])
def test_random_sensor_project_seed_determinism(mode: str):
    cfg = _config()
    spectrum = np.linspace(0.1, 1.0, cfg.highres_axis_nm.shape[0])
    batch = [
        (spectrum, cfg.highres_axis_nm),
        (spectrum * 0.2, cfg.highres_axis_nm),
    ]

    aug1 = RandomSensorProject(cfg, mode=mode)
    aug2 = RandomSensorProject(cfg, mode=mode)

    out1 = aug1(batch)
    out2 = aug2(batch)

    assert len(out1) == len(out2) == len(batch)
    for proj1, proj2 in zip(out1, out2, strict=True):
        assert np.allclose(proj1.values, proj2.values)
        assert np.allclose(proj1.centers_nm, proj2.centers_nm)
        assert np.allclose(proj1.srf_matrix, proj2.srf_matrix)


def test_random_sensor_project_per_batch_shares_sensor():
    cfg = _config()
    values = 0.5 + 0.5 * np.sin(cfg.highres_axis_nm / 500.0)
    batch = [
        (values, cfg.highres_axis_nm),
        (values * 1.1, cfg.highres_axis_nm),
    ]

    transform = RandomSensorProject(cfg, mode="per_batch")
    projected = transform(batch)

    assert len(projected) == 2
    assert np.allclose(projected[0].centers_nm, projected[1].centers_nm)
    assert np.allclose(projected[0].srf_matrix, projected[1].srf_matrix)


def test_random_sensor_project_fixed_sensor_across_calls():
    cfg = _config()
    values = 0.5 + 0.5 * np.sin(cfg.highres_axis_nm / 500.0)
    batch = [(values, cfg.highres_axis_nm)]

    transform = RandomSensorProject(cfg, mode="fixed")
    first = transform(batch)[0]
    second = transform(batch)[0]

    assert np.allclose(first.centers_nm, second.centers_nm)
    assert np.allclose(first.srf_matrix, second.srf_matrix)


def test_random_sensor_project_output_shapes():
    cfg = _config()
    values = 0.5 + 0.5 * np.sin(cfg.highres_axis_nm / 500.0)
    batch = [(values, cfg.highres_axis_nm)]

    transform = RandomSensorProject(cfg)
    projected = transform(batch)[0]

    assert projected.values.shape[0] == cfg.n_bands
    assert projected.srf_matrix.shape[0] == cfg.n_bands
    assert np.all(np.diff(projected.centers_nm) > 0)
