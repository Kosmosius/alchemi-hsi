from __future__ import annotations

import numpy as np
import xarray as xr

from alchemi_hsi.io.avirisng import avirisng_pixel, load_avirisng_l1b


def _write_avirisng_file(
    path,
    wavelengths_um,
    radiance_value: float = 1.0,
    units: str = "microwatts per square centimeter per steradian per nanometer",
    fwhm_um=None,
    bad_bands=None,
    band_mask=None,
):
    wavelengths_um = np.asarray(wavelengths_um, dtype=np.float64)
    band_count = wavelengths_um.shape[0]

    data = xr.Dataset()
    data["radiance"] = xr.DataArray(
        np.full((band_count, 3, 2), radiance_value, dtype=np.float64),
        dims=("band", "y", "x"),
        attrs={"units": units},
    )
    data["wavelength"] = xr.DataArray(
        wavelengths_um,
        dims=("band",),
        attrs={"units": "micrometers"},
    )

    if fwhm_um is not None:
        data["fwhm"] = xr.DataArray(
            np.asarray(fwhm_um, dtype=np.float64),
            dims=("band",),
            attrs={"units": "micrometers"},
        )

    if bad_bands is not None:
        data["bad_band_list"] = xr.DataArray(
            np.asarray(bad_bands, dtype=np.int32),
            dims=("index",),
        )

    if band_mask is not None:
        data["band_mask"] = xr.DataArray(
            np.asarray(band_mask, dtype=np.int8),
            dims=("band",),
        )

    data.to_netcdf(path)


def test_wavelengths_monotonic_avirisng(tmp_path):
    path = tmp_path / "avirisng.nc"
    wavelengths_um = np.array([0.380, 1.650, 0.760, 2.510], dtype=np.float64)
    fwhm = np.full(wavelengths_um.shape[0], 0.010, dtype=np.float64)
    _write_avirisng_file(path, wavelengths_um, fwhm_um=fwhm)

    ds = load_avirisng_l1b(path)

    wavelengths = ds["wavelength_nm"].values
    assert np.all(np.diff(wavelengths) > 0)
    assert 380 <= wavelengths[0] < wavelengths[-1] <= 2510
    assert ds.attrs["units"] == "W·m^-2·sr^-1·nm^-1"
    assert ds.attrs["quantity"] == "radiance"
    np.testing.assert_allclose(ds["fwhm_nm"].values, np.sort(fwhm) * 1_000.0)


def test_units_conversion_nm_avirisng(tmp_path):
    path = tmp_path / "avirisng_units.nc"
    wavelengths_um = np.array([0.400, 1.000, 2.000], dtype=np.float64)
    _write_avirisng_file(path, wavelengths_um, radiance_value=150.0)

    ds = load_avirisng_l1b(path)

    pixel = ds["radiance"].sel(y=0, x=0).values
    expected = np.full(3, 150.0e-6 * 1e4, dtype=np.float64)
    np.testing.assert_allclose(pixel, expected)
    assert ds.attrs["units"] == "W·m^-2·sr^-1·nm^-1"


def test_mask_water_bands_avirisng(tmp_path):
    path = tmp_path / "avirisng_mask.nc"
    wavelengths_um = np.array([0.8, 1.38, 1.45, 1.93, 2.20], dtype=np.float64)
    band_mask = [1, 1, 0, 1, 1]
    bad_bands = [2]
    _write_avirisng_file(path, wavelengths_um, band_mask=band_mask, bad_bands=bad_bands)

    ds = load_avirisng_l1b(path)

    mask = ds["band_mask"].values
    assert mask.shape[0] == ds.sizes["band"]
    assert mask.dtype == bool

    expected = np.array([True, False, False, False, True], dtype=bool)
    np.testing.assert_array_equal(mask, expected)

    spec = avirisng_pixel(ds, 1, 1)
    assert spec.mask is not None
    np.testing.assert_array_equal(spec.mask, mask)
    np.testing.assert_allclose(spec.wavelengths.nm, ds["wavelength_nm"].values)
    assert spec.meta["sensor"] == "avirisng"
