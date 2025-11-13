from __future__ import annotations

import numpy as np

import xarray as xr
from alchemi_hsi.io.enmap import enmap_pixel, load_enmap_l1b


def _write_slice(path, wavelengths_um, radiance_value=1.0, units="W m-2 sr-1 um-1", mask=None):
    wavelengths_um = np.asarray(wavelengths_um, dtype=np.float64)
    band_count = wavelengths_um.shape[0]
    radiance = np.full((2, 3, band_count), radiance_value, dtype=np.float64)
    fwhm = np.full(band_count, 0.012, dtype=np.float64)

    data = xr.Dataset(
        {
            "radiance": (
                ("rows", "cols", "spectral_band"),
                radiance,
                {"units": units},
            ),
            "bandwidth": (("spectral_band",), fwhm, {"units": "um"}),
        },
        coords={
            "spectral_band": ("spectral_band", wavelengths_um, {"units": "um"}),
        },
    )
    if mask is not None:
        data["quality_flags"] = (("spectral_band",), np.asarray(mask))

    data.to_netcdf(path)


def test_wavelengths_monotonic_enmap(tmp_path):
    vnir_path = tmp_path / "vnir.nc"
    swir_path = tmp_path / "swir.nc"

    _write_slice(vnir_path, [0.420, 0.430, 0.440, 0.450], units="W m-2 sr-1 um-1")
    _write_slice(swir_path, [0.97, 1.10, 2.45], units="W m-2 sr-1 um-1")

    ds = load_enmap_l1b(vnir_path, swir_path)

    wavelengths = ds["wavelength_nm"].values
    assert np.all(np.diff(wavelengths) > 0)
    assert 420 <= wavelengths[0] < wavelengths[-1] <= 2450
    assert ds.attrs["units"] == "W·m^-2·sr^-1·nm^-1"
    assert ds.attrs["quantity"] == "radiance"


def test_units_conversion_nm_enmap(tmp_path):
    vnir_path = tmp_path / "vnir.nc"
    swir_path = tmp_path / "swir.nc"

    _write_slice(vnir_path, [0.420, 0.430], radiance_value=2.0, units="mW m-2 sr-1 um-1")
    _write_slice(swir_path, [1.0, 1.2], radiance_value=5.0, units="W m-2 sr-1 nm-1")

    ds = load_enmap_l1b(vnir_path, swir_path)

    pixel = ds["radiance"].sel(y=0, x=0).values
    expected = np.concatenate([
        np.full(2, 2.0e-6, dtype=np.float64),
        np.full(2, 5.0, dtype=np.float64),
    ])
    np.testing.assert_allclose(pixel, expected)
    assert ds.attrs["units"] == "W·m^-2·sr^-1·nm^-1"


def test_mask_water_bands_enmap(tmp_path):
    vnir_path = tmp_path / "vnir.nc"
    swir_path = tmp_path / "swir.nc"

    _write_slice(
        vnir_path,
        [0.420, 0.430, 0.440],
        mask=[1, 0, 1],
        units="W m-2 sr-1 um-1",
    )
    _write_slice(
        swir_path,
        [1.00, 1.20],
        mask=[0, 1],
        units="W m-2 sr-1 um-1",
    )

    ds = load_enmap_l1b(vnir_path, swir_path)

    mask = ds["band_mask"].values
    assert mask.shape[0] == ds.sizes["band"]
    assert mask.dtype == bool

    spec = enmap_pixel(ds, 1, 2)
    assert spec.mask is not None
    np.testing.assert_array_equal(spec.mask, mask)
    np.testing.assert_allclose(spec.wavelengths.nm, ds["wavelength_nm"].values)
    assert spec.meta["sensor"] == "enmap"
    assert "fwhm_nm" in spec.meta
