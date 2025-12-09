from __future__ import annotations

import numpy as np
import xarray as xr

from alchemi.data.adapters import (
    iter_aviris_ng_pixels,
    iter_aviris_ng_reflectance_pixels,
)
from alchemi.data.io import load_avirisng_l1b
from alchemi.registry import srfs


def _write_l1b(path, radiance_value: float, band_mask: np.ndarray | None = None) -> None:
    centers = np.asarray(srfs.get_srf("aviris-ng").centers_nm, dtype=np.float64)
    band_count = centers.size
    radiance = np.full((1, 1, band_count), radiance_value, dtype=np.float64)

    ds = xr.Dataset()
    ds["radiance"] = xr.DataArray(
        radiance,
        dims=("y", "x", "band"),
        attrs={"units": "microwatts per square centimeter per steradian per nanometer"},
    )
    ds["wavelength"] = xr.DataArray(
        centers / 1_000.0, dims=("band",), attrs={"units": "micrometers"}
    )

    if band_mask is not None:
        ds["band_mask"] = xr.DataArray(np.asarray(band_mask, dtype=np.int8), dims=("band",))

    ds.to_netcdf(path)


def _write_reflectance(path, values: np.ndarray) -> None:
    band_count = values.shape[-1]
    centers = np.asarray(srfs.get_srf("aviris-ng").centers_nm[:band_count], dtype=np.float64)
    ds = xr.Dataset()
    ds["reflectance"] = xr.DataArray(values, dims=("y", "x", "band"))
    ds = ds.assign_coords(wavelength_nm=("band", centers))
    ds.to_netcdf(path)


def test_radiance_samples_use_loader_units(tmp_path) -> None:
    path = tmp_path / "avirisng.nc"
    centers = np.asarray(srfs.get_srf("aviris-ng").centers_nm, dtype=np.float64)
    user_mask = np.ones_like(centers, dtype=bool)
    user_mask[10] = False
    _write_l1b(path, radiance_value=150.0, band_mask=user_mask)

    sample = next(iter_aviris_ng_pixels(str(path)))
    ds = load_avirisng_l1b(path)

    assert np.all(np.diff(sample.spectrum.wavelength_nm) > 0)
    expected = ds["radiance"].sel(y=0, x=0).values
    np.testing.assert_allclose(sample.spectrum.values, expected)

    mask = sample.quality_masks["valid_band"]
    assert mask.shape == centers.shape
    assert not mask[10]
    assert np.all(mask <= 1)
    np.testing.assert_array_equal(sample.band_meta.valid_mask, mask)
    assert sample.band_meta.srf_source[0] == "official"

    assert sample.srf_matrix is not None
    assert (
        sample.srf_matrix.matrix.shape[0] == sample.srf_matrix.matrix.shape[1] == centers.shape[0]
    )


def test_reflectance_samples_match_radiance_grid(tmp_path) -> None:
    path = tmp_path / "avirisng_reflectance.nc"
    centers = np.asarray(srfs.get_srf("aviris-ng").centers_nm[:5], dtype=np.float64)
    reflectance = np.linspace(0, 1, centers.shape[0], dtype=np.float64)
    values = reflectance.reshape(1, 1, -1)
    _write_reflectance(path, values)

    sample = next(iter_aviris_ng_reflectance_pixels(str(path)))

    np.testing.assert_array_equal(sample.spectrum.wavelength_nm, centers)
    np.testing.assert_array_equal(sample.spectrum.values, reflectance)
    assert sample.spectrum.kind == "reflectance"

    mask = sample.quality_masks["valid_band"]
    assert mask.dtype == bool
    assert mask.shape == centers.shape
    assert np.all((sample.spectrum.values >= 0.0) & (sample.spectrum.values <= 1.0))
