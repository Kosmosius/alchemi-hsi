"""End-to-end integration test for the AVIRIS-NG ingest adapter."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from alchemi.data.io import load_avirisng_l1b
from alchemi.ingest.avirisng import from_avirisng_l1b
from alchemi.physics.rad_reflectance import radiance_to_toa_reflectance
from alchemi.physics.solar import esun_for_sample
from alchemi.registry import get_srf

pytestmark = pytest.mark.physics_and_metadata


def _write_aviris_fixture(path):
    wavelengths = np.array([410.0, 870.0, 1050.0, 2100.0], dtype=np.float64)
    radiance = (
        np.arange(2 * 2 * wavelengths.size, dtype=np.float64).reshape(wavelengths.size, 2, 2)
        + 5.0
    )

    ds = xr.Dataset(
        {
            "radiance": (("band", "y", "x"), radiance, {"units": "W m^-2 sr^-1 nm^-1"}),
            "band_mask": (("band",), np.array([1, 0, 1, 1], dtype=np.int8)),
        },
        coords={"wavelength": ("band", wavelengths, {"units": "nanometers"})},
    )
    ds.to_netcdf(path)
    return wavelengths


def test_avirisng_ingest_respects_physics_and_metadata(tmp_path):
    path = tmp_path / "aviris.nc"
    wavelengths = _write_aviris_fixture(path)

    dataset = load_avirisng_l1b(path)
    cube = from_avirisng_l1b(dataset)

    height, width, band_count = cube.shape
    assert height > 0 and width > 0
    assert band_count == cube.axis.shape[0] == wavelengths.shape[0]
    assert np.all(np.diff(cube.axis) > 0)
    assert cube.value_kind.value == "radiance"

    assert cube.srf_id is not None
    srf = get_srf(str(cube.srf_id))
    assert srf.centers_nm.size >= band_count

    assert cube.band_mask is not None
    np.testing.assert_array_equal(cube.band_mask, dataset["band_mask"].values)

    sample = cube.sample_at(0, 0)
    np.testing.assert_array_equal(sample.spectrum.mask, cube.band_mask)

    esun_band = esun_for_sample(sample, mode="interp")
    reflectance = radiance_to_toa_reflectance(
        sample.spectrum, esun_band=esun_band, d_au=1.0, solar_zenith_deg=30.0
    )
    valid_reflectance = reflectance.values[cube.band_mask]
    p1, p99 = np.nanpercentile(valid_reflectance, [1, 99])
    assert p1 >= 0.0
    assert p99 <= 1.2

