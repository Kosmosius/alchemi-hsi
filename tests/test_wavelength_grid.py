import numpy as np
import pytest

from alchemi.types import (
    WAVELENGTH_GRID_DUPLICATE_EPS,
    WAVELENGTH_GRID_MONOTONICITY_EPS,
    WavelengthGrid,
)


def test_strictly_increasing_grid_ok() -> None:
    nm = np.array([400.0, 401.0, 402.0], dtype=np.float64)
    grid = WavelengthGrid(nm)
    np.testing.assert_allclose(grid.nm, nm)


def test_small_noise_grid_ok() -> None:
    nm = np.array([400.0, 400.0000000001, 400.0000000002], dtype=np.float64)
    grid = WavelengthGrid(nm)
    np.testing.assert_allclose(grid.nm, nm)


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
    nm = np.array(
        [400.0, 400.0000000005, 400.0000000004],
        dtype=np.float64,
    )
    assert np.all(np.diff(nm) > -WAVELENGTH_GRID_MONOTONICITY_EPS)
    grid = WavelengthGrid(nm)
    np.testing.assert_allclose(grid.nm, nm)
