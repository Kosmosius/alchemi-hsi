"""End-to-end integration test for the EnMAP ingest adapter."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from alchemi.data.io import load_enmap_l1b
from alchemi.ingest.enmap import from_enmap_l1b
from alchemi.physics.rad_reflectance import radiance_to_toa_reflectance
from alchemi.physics.solar import esun_for_sample
from alchemi.registry import get_srf

pytestmark = pytest.mark.physics_and_metadata


def _write_enmap_slice(path, wavelengths_um, radiance_value, units, mask=None):
    wavelengths_um = np.asarray(wavelengths_um, dtype=np.float64)
    band_count = wavelengths_um.size
    radiance = np.full((2, 3, band_count), radiance_value, dtype=np.float64)
    fwhm = np.full(band_count, 0.010, dtype=np.float64)

    data = xr.Dataset(
        {
            "radiance": (("rows", "cols", "spectral_band"), radiance, {"units": units}),
            "bandwidth": (("spectral_band",), fwhm, {"units": "um"}),
        },
        coords={"spectral_band": ("spectral_band", wavelengths_um, {"units": "um"})},
    )
    if mask is not None:
        data["quality_flags"] = (("spectral_band",), np.asarray(mask))

    data.to_netcdf(path)


def test_enmap_ingest_respects_physics_and_metadata(tmp_path):
    vnir_path = tmp_path / "vnir.nc"
    swir_path = tmp_path / "swir.nc"
    _write_enmap_slice(
        vnir_path, [0.45, 0.55, 0.65], radiance_value=1.5, units="W m-2 sr-1 um-1", mask=[1, 0, 1]
    )
    _write_enmap_slice(
        swir_path, [1.20, 1.40], radiance_value=4.0, units="W m-2 sr-1 nm-1", mask=[0, 1]
    )

    dataset = load_enmap_l1b(vnir_path, swir_path)
    cube = from_enmap_l1b(dataset)

    height, width, band_count = cube.shape
    assert height > 0 and width > 0
    assert band_count == cube.axis.shape[0] == dataset["wavelength_nm"].shape[0]
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
