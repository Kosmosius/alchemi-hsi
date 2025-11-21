from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from alchemi.data.io import (
    HYTES_BAND_COUNT,
    HYTES_WAVELENGTHS_NM,
    load_avirisng_l1b,
    load_emit_l1b,
    load_enmap_l1b,
    load_hytes_l1b_bt,
)


rasterio = pytest.importorskip("rasterio")


def _write_emit_fixture(path):
    width, height, bands = 4, 3, 5
    wavelengths_um = np.array([0.42, 0.65, 1.42, 1.93, 2.25], dtype=np.float32)
    radiance_um = np.arange(bands * height * width, dtype=np.float32).reshape(bands, height, width) + 1.0

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=bands,
        dtype=radiance_um.dtype,
    ) as dst:
        dst.write(radiance_um)
        dst.update_tags(
            wavelengths=",".join(map(str, wavelengths_um)),
            wavelength_units="micrometers",
            radiance_units="W/m^2/sr/um",
        )

    return wavelengths_um, radiance_um


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


def _write_aviris_fixture(path):
    wavelengths = np.array([410.0, 870.0, 1050.0, 2100.0], dtype=np.float64)
    radiance = np.arange(2 * 2 * wavelengths.size, dtype=np.float64).reshape(wavelengths.size, 2, 2) + 5.0

    ds = xr.Dataset(
        {
            "radiance": (("band", "y", "x"), radiance, {"units": "W m^-2 sr^-1 nm^-1"}),
            "band_mask": (("band",), np.array([1, 0, 1, 1], dtype=np.int8)),
        },
        coords={"wavelength": ("band", wavelengths, {"units": "nanometers"})},
    )
    ds.to_netcdf(path)
    return wavelengths, radiance


def _write_hytes_fixture(path):
    data = np.linspace(280.0, 320.0, num=HYTES_BAND_COUNT * 2 * 2, dtype=np.float64).reshape(HYTES_BAND_COUNT, 2, 2)

    raw = xr.Dataset(
        {"BrightnessTemperature": (("band", "y", "x"), data, {"units": "K"})},
        coords={"band": np.arange(HYTES_BAND_COUNT, dtype=np.int32)},
    )
    raw.attrs["bt_units"] = "K"
    raw.to_netcdf(path)
    return data.transpose(1, 2, 0)


def test_emit_golden_ingest(tmp_path):
    path = tmp_path / "emit_fixture.tif"
    wavelengths_um, radiance_um = _write_emit_fixture(path)

    ds = load_emit_l1b(path)
    assert ds["radiance"].shape == (3, 4, 5)
    np.testing.assert_allclose(ds.coords["wavelength_nm"], np.sort(wavelengths_um) * 1000.0)
    expected = np.moveaxis(radiance_um / 1000.0, 0, -1)
    np.testing.assert_allclose(ds["radiance"].values, expected)
    assert ds.attrs["sensor"].lower() == "emit"
    assert ds["band_mask"].dtype == bool
    assert not ds["band_mask"].values[2]


def test_enmap_golden_ingest(tmp_path):
    vnir_path = tmp_path / "vnir.nc"
    swir_path = tmp_path / "swir.nc"
    _write_enmap_slice(vnir_path, [0.45, 0.55, 0.65], radiance_value=1.5, units="W m-2 sr-1 um-1", mask=[1, 0, 1])
    _write_enmap_slice(swir_path, [1.20, 1.40], radiance_value=4.0, units="W m-2 sr-1 nm-1", mask=[0, 1])

    ds = load_enmap_l1b(vnir_path, swir_path)
    assert ds["radiance"].shape == (2, 3, 5)
    assert ds.attrs["units"] == "W·m^-2·sr^-1·nm^-1"
    pixel = ds["radiance"].isel(y=0, x=0).values
    expected = np.concatenate([np.full(3, 1.5e-3), np.full(2, 4.0)])
    np.testing.assert_allclose(pixel, expected)
    assert ds["band_mask"].values.tolist() == [True, False, True, False, True]


def test_avirisng_golden_ingest(tmp_path):
    path = tmp_path / "aviris.nc"
    wavelengths_um, radiance = _write_aviris_fixture(path)

    ds = load_avirisng_l1b(path)
    assert ds["radiance"].shape == (2, 2, wavelengths_um.size)
    assert ds.attrs["units"] == "W·m^-2·sr^-1·nm^-1"
    np.testing.assert_allclose(ds["radiance"].isel(y=0, x=0), radiance[:, 0, 0])
    assert ds["band_mask"].values.tolist() == [True, False, True, True]


def test_hytes_golden_ingest(tmp_path):
    path = tmp_path / "hytes.nc"
    expected_bt = _write_hytes_fixture(path)

    ds = load_hytes_l1b_bt(path)
    assert ds["brightness_temp"].shape == (2, 2, expected_bt.shape[-1])
    np.testing.assert_allclose(ds["brightness_temp"].values, expected_bt)
    np.testing.assert_allclose(ds["wavelength_nm"].values, HYTES_WAVELENGTHS_NM)
    assert ds.attrs["sensor"].lower() == "hytes"
