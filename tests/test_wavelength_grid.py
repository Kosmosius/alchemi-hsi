import numpy as np
import pytest

from alchemi.types import WAVELENGTH_GRID_DUPLICATE_EPS, WavelengthGrid


def test_strictly_increasing_grid_ok() -> None:
    nm = np.array([400.0, 401.0, 402.0], dtype=np.float64)
    grid = WavelengthGrid(nm)
    np.testing.assert_allclose(grid.nm, nm)


def test_small_noise_grid_ok() -> None:
    nm = np.array([400.0, 400.0000000001, 400.0000000002], dtype=np.float64)
    grid = WavelengthGrid(nm)
    np.testing.assert_allclose(grid.nm, nm)


def test_non_1d_grid_fails() -> None:
    nm = np.array([[400.0, 401.0], [402.0, 403.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="1-D"):
        WavelengthGrid(nm)


def test_decreasing_grid_fails() -> None:
    nm = np.array([400.0, 399.9, 402.0], dtype=np.float64)
    with pytest.raises(ValueError):
        WavelengthGrid(nm)


def test_exact_duplicates_fail() -> None:
    nm = np.array([400.0, 400.0, 401.0], dtype=np.float64)
    with pytest.raises(ValueError):
        WavelengthGrid(nm)


def test_near_duplicates_fail_within_tolerance() -> None:
    nm = np.array(
        [400.0, 400.0 + 0.5 * WAVELENGTH_GRID_DUPLICATE_EPS, 401.0],
        dtype=np.float64,
    )
    with pytest.raises(ValueError):
        WavelengthGrid(nm)


def test_small_negative_jitter_ok() -> None:
    nm = np.array([400.0, 400.0000000005, 400.0000000004], dtype=np.float64)
    with pytest.raises(ValueError, match="strictly increasing"):
        WavelengthGrid(nm)


def test_from_any_converts_microns() -> None:
    microns = np.array([0.4, 0.41, 0.42], dtype=np.float64)
    grid = WavelengthGrid.from_any(microns, units="um")
    np.testing.assert_allclose(grid.nm, microns * 1e3)


def test_from_any_rejects_unknown_units() -> None:
    nm = np.array([400.0, 401.0], dtype=np.float64)
    with pytest.raises(ValueError):
        WavelengthGrid.from_any(nm, units="hz")
