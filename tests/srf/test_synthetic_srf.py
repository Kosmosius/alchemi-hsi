import numpy as np

try:
    _integrate = np.trapezoid
except AttributeError:  # pragma: no cover - NumPy < 2.0 fallback
    _integrate = np.trapz

from alchemi.srf import project_to_sensor, rand_srf_grid


def test_rand_srf_grid_rows_normalized():
    grid = np.linspace(400.0, 2500.0, 1024)
    _, matrix = rand_srf_grid(
        grid,
        n_bands=12,
        center_jitter_nm=5.0,
        fwhm_range_nm=(8.0, 20.0),
        shape="gaussian",
        seed=123,
    )

    for wl, resp in zip(matrix.bands_nm, matrix.bands_resp, strict=True):
        area = float(_integrate(resp, wl))
        assert abs(area - 1.0) < 1e-6


def test_flat_spectrum_projection_preserves_mean():
    grid = np.linspace(400.0, 2500.0, 2048)
    spectrum = np.full(grid.shape[0], 0.75, dtype=np.float64)
    _, matrix = rand_srf_grid(
        grid,
        n_bands=16,
        center_jitter_nm=2.5,
        fwhm_range_nm=(10.0, 30.0),
        shape="hamming",
        seed=321,
    )

    projected = project_to_sensor(grid, spectrum, matrix.centers_nm, srf=matrix)
    assert projected.shape[0] == matrix.centers_nm.shape[0]
    assert np.allclose(projected, spectrum[0], atol=1e-6)
