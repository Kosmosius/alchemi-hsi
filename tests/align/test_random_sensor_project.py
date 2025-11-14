import numpy as np

from alchemi.align import RandomSensorProject


def test_random_sensor_project_seed_determinism():
    grid = np.linspace(400.0, 2500.0, 1500)
    spectrum = np.linspace(0.1, 1.0, grid.size)

    aug1 = RandomSensorProject(
        grid,
        n_bands=32,
        center_jitter_nm=6.0,
        fwhm_range_nm=(12.0, 24.0),
        shape="gaussian",
        seed=2024,
    )
    aug2 = RandomSensorProject(
        grid,
        n_bands=32,
        center_jitter_nm=6.0,
        fwhm_range_nm=(12.0, 24.0),
        shape="gaussian",
        seed=2024,
    )

    out1 = aug1(spectrum)
    out2 = aug2(spectrum)

    for arr1, arr2 in zip(out1, out2, strict=True):
        assert np.allclose(arr1, arr2)


def test_random_sensor_project_band_center_sanity():
    grid = np.linspace(450.0, 2450.0, 1800)
    spectrum = np.sin(grid / 1800.0)
    transform = RandomSensorProject(
        grid,
        n_bands=24,
        center_jitter_nm=4.0,
        fwhm_range_nm=(8.0, 18.0),
        shape="box",
        seed=77,
    )

    projected, centers, fwhm = transform(spectrum)

    assert projected.shape[0] == centers.shape[0] == fwhm.shape[0]
    assert np.all(np.diff(centers) > 0)
    assert centers[0] >= grid[0] - 1e-6
    assert centers[-1] <= grid[-1] + 1e-6
    assert np.all(fwhm > 0)
