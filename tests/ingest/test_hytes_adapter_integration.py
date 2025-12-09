"""End-to-end integration test for the HyTES ingest adapter."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from alchemi.data.io import HYTES_BAND_COUNT, load_hytes_l1b_bt
from alchemi.ingest.hytes import from_hytes_bt
from alchemi.registry import get_srf

pytestmark = pytest.mark.physics_and_metadata


def _write_hytes_fixture(path):
    data = np.linspace(280.0, 320.0, num=HYTES_BAND_COUNT * 2 * 2, dtype=np.float64).reshape(
        HYTES_BAND_COUNT, 2, 2
    )

    raw = xr.Dataset(
        {"BrightnessTemperature": (("band", "y", "x"), data, {"units": "K"})},
        coords={"band": np.arange(HYTES_BAND_COUNT, dtype=np.int32)},
    )
    raw.attrs["bt_units"] = "K"
    raw.to_netcdf(path)


def test_hytes_ingest_respects_physics_and_metadata(tmp_path):
    path = tmp_path / "hytes.nc"
    _write_hytes_fixture(path)

    dataset = load_hytes_l1b_bt(path)
    cube = from_hytes_bt(dataset)

    height, width, band_count = cube.shape
    assert height > 0 and width > 0
    assert band_count == cube.axis.shape[0]
    assert np.all(np.diff(cube.axis) > 0)
    assert cube.value_kind.value in {"brightness_temperature", "brightness_temp"}

    assert cube.srf_id is not None
    srf = get_srf(str(cube.srf_id))
    assert srf.centers_nm.size >= band_count

    assert cube.band_mask is not None
    np.testing.assert_array_equal(cube.band_mask, dataset["band_mask"].values)

    sample = cube.sample_at(0, 0)
    np.testing.assert_array_equal(sample.spectrum.mask, cube.band_mask)

    valid_bt = sample.spectrum.values[cube.band_mask]
    p1, p99 = np.nanpercentile(valid_bt, [1, 99])
    assert 150.0 <= p1 <= 400.0
    assert 150.0 <= p99 <= 400.0
