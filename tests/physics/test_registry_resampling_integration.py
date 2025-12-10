"""Integration tests combining registry SRFs with resampling utilities."""

from __future__ import annotations

import numpy as np
import pytest

from alchemi.physics.resampling import convolve_to_bands
from alchemi.registry import srfs
from alchemi.types import Spectrum, SRFMatrix, WavelengthGrid

from tests.spectral.test_srf_validation_and_normalization import (
    _registered_sensor_ids,
)


@pytest.mark.parametrize("sensor_id", _registered_sensor_ids())
def test_flat_lab_spectrum_remains_flat_for_registered_srfs(sensor_id: str) -> None:
    srf = srfs.get_srf(sensor_id)

    union_grid = np.unique(np.concatenate([np.asarray(nm, dtype=np.float64) for nm in srf.bands_nm]))
    flat_value = 3.21
    high_res = Spectrum.from_radiance(
        WavelengthGrid(union_grid), np.full_like(union_grid, flat_value, dtype=np.float64)
    )

    banded = convolve_to_bands(high_res, srf)
    mask = banded.mask if banded.mask is not None else np.zeros_like(banded.values, dtype=bool)

    np.testing.assert_allclose(banded.values[~mask], flat_value, rtol=1e-3, atol=1e-6)


def test_resampling_with_interpolation_remains_non_negative() -> None:
    sensor_id = "emit"
    srf = srfs.get_srf(sensor_id)

    all_nm = np.concatenate(srf.bands_nm)
    grid = np.linspace(all_nm.min(), all_nm.max(), 500)
    values = 0.1 + 0.001 * (grid - grid.min())
    high_res = Spectrum.from_radiance(WavelengthGrid(grid), values)

    banded = convolve_to_bands(high_res, srf)
    mask = banded.mask if banded.mask is not None else np.zeros_like(banded.values, dtype=bool)
    valid = banded.values[~mask]

    assert np.all(np.isfinite(valid))
    assert np.all(valid >= values.min() - 1e-9)
    assert np.all(valid <= values.max() + 1e-6)


def test_zero_area_srf_row_raises() -> None:
    nm = np.linspace(400.0, 500.0, 5)
    valid_band = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    zero_band = np.zeros_like(nm)
    srf = SRFMatrix(
        sensor="synthetic",
        centers_nm=np.array([425.0, 475.0]),
        bands_nm=[nm, nm],
        bands_resp=[valid_band, zero_band],
    )

    flat_high_res = Spectrum.from_radiance(WavelengthGrid(nm), np.ones_like(nm))

    with pytest.raises(ValueError):
        convolve_to_bands(flat_high_res, srf)
