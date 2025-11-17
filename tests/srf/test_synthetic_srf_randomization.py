import numpy as np

from alchemi.srf.synthetic import (
    SyntheticSensorConfig,
    project_lab_to_synthetic,
    rand_srf_grid,
)

try:  # NumPy >= 2.0
    from numpy import trapezoid as _integrate
except ImportError:  # pragma: no cover - fallback for NumPy < 2.0
    from numpy import trapz as _integrate  # type: ignore[attr-defined]


def test_rand_srf_grid_rows_normalized():
    highres = np.linspace(400.0, 2500.0, 1024)
    centers, matrix = rand_srf_grid(
        highres,
        n_bands=12,
        center_jitter_nm=5.0,
        fwhm_range_nm=(8.0, 20.0),
        shape="gaussian",
        seed=123,
    )

    assert centers.shape[0] == matrix.shape[0]
    for row in matrix:
        area = _integrate(row, highres)
        assert np.isclose(area, 1.0, atol=1e-6)


def test_flat_spectrum_projection_preserves_mean():
    highres = np.linspace(400.0, 2500.0, 2048)
    spectrum = np.full(highres.shape[0], 0.75, dtype=np.float64)
    cfg = SyntheticSensorConfig(
        highres_axis_nm=highres,
        n_bands=16,
        center_jitter_nm=2.5,
        fwhm_range_nm=(10.0, 30.0),
        shape="box",
        seed=321,
    )

    projected = project_lab_to_synthetic(spectrum, highres, cfg)
    assert projected.values.shape[0] == cfg.n_bands
    assert np.allclose(projected.values, spectrum[0], atol=1e-3)


def test_rand_srf_grid_seed_determinism():
    highres = np.linspace(450.0, 2450.0, 1500)
    centers1, matrix1 = rand_srf_grid(
        highres,
        n_bands=20,
        center_jitter_nm=4.0,
        fwhm_range_nm=(6.0, 12.0),
        shape="hamming",
        seed=2024,
    )
    centers2, matrix2 = rand_srf_grid(
        highres,
        n_bands=20,
        center_jitter_nm=4.0,
        fwhm_range_nm=(6.0, 12.0),
        shape="hamming",
        seed=2024,
    )

    assert np.allclose(centers1, centers2)
    assert np.allclose(matrix1, matrix2)
