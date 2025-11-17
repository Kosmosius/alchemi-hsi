import numpy as np

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


def test_random_sensor_project_seed_determinism():
    cfg = _config()
    spectrum = np.linspace(0.1, 1.0, cfg.highres_axis_nm.shape[0])
    batch = [(spectrum, cfg.highres_axis_nm)]

    aug1 = RandomSensorProject(cfg)
    aug2 = RandomSensorProject(cfg)

    out1 = aug1(batch)
    out2 = aug2(batch)

    assert len(out1) == len(out2) == 1
    proj1, proj2 = out1[0], out2[0]
    assert np.allclose(proj1.values, proj2.values)
    assert np.allclose(proj1.centers_nm, proj2.centers_nm)
    assert np.allclose(proj1.srf_matrix, proj2.srf_matrix)


def test_random_sensor_project_output_shapes():
    cfg = _config()
    values = np.sin(cfg.highres_axis_nm / 500.0)
    batch = [(values, cfg.highres_axis_nm)]

    transform = RandomSensorProject(cfg)
    projected = transform(batch)[0]

    assert projected.values.shape[0] == cfg.n_bands
    assert projected.srf_matrix.shape[0] == cfg.n_bands
    assert np.all(np.diff(projected.centers_nm) > 0)
