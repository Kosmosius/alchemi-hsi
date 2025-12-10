import numpy as np
import pytest
import xarray as xr

from alchemi.data.adapters.aviris_ng import iter_aviris_ng_pixels
from alchemi.data.adapters.emit import iter_emit_pixels
from alchemi.data.adapters.enmap import iter_enmap_pixels
from alchemi.data.adapters.hytes import iter_hytes_pixels
from alchemi.physics.rad_reflectance import radiance_sample_to_toa_reflectance
from alchemi.spectral import ViewingGeometry
from alchemi.spectral.sample import GeoMeta


def _make_grid_dataset(variable: str, *, add_geo: bool = True) -> xr.Dataset:
    y = np.arange(2)
    x = np.arange(2)
    band = np.arange(3)
    base_val = 300.0 if "temp" in variable else 0.01
    data = np.full((2, 2, 3), base_val, dtype=np.float64)
    ds = xr.Dataset({variable: (("y", "x", "band"), data)})
    ds = ds.assign_coords(wavelength_nm=("band", np.array([500.0, 600.0, 700.0])))
    if add_geo:
        lat_grid = xr.DataArray(np.full((2, 2), 10.0), dims=("y", "x"))
        lon_grid = xr.DataArray(np.full((2, 2), 20.0), dims=("y", "x"))
        ds = ds.assign_coords(latitude=lat_grid, longitude=lon_grid)
    geometry_vals = np.array([[30.0, 31.0], [32.0, 33.0]], dtype=np.float64)
    ds["solar_zenith"] = xr.DataArray(geometry_vals, dims=("y", "x"))
    ds["solar_azimuth"] = xr.DataArray(np.full((2, 2), 120.0), dims=("y", "x"))
    ds["view_zenith"] = xr.DataArray(np.full((2, 2), 10.0), dims=("y", "x"))
    ds["view_azimuth"] = xr.DataArray(np.full((2, 2), 45.0), dims=("y", "x"))
    ds["earth_sun_distance_au"] = xr.DataArray(np.full((2, 2), 1.01), dims=("y", "x"))
    return ds


def _assert_geometry(sample):
    assert sample.viewing_geometry is not None
    assert isinstance(sample.viewing_geometry, ViewingGeometry)
    assert pytest.approx(30.0) == float(sample.viewing_geometry.solar_zenith_deg)
    assert pytest.approx(120.0) == float(sample.viewing_geometry.solar_azimuth_deg)
    assert pytest.approx(10.0) == float(sample.viewing_geometry.view_zenith_deg)
    assert pytest.approx(45.0) == float(sample.viewing_geometry.view_azimuth_deg)
    assert pytest.approx(1.01) == float(sample.viewing_geometry.earth_sun_distance_au)


def _assert_geo(sample):
    assert sample.geo is not None
    assert isinstance(sample.geo, GeoMeta)
    assert pytest.approx(10.0) == float(sample.geo.lat)
    assert pytest.approx(20.0) == float(sample.geo.lon)


def test_emit_geometry_and_geo(monkeypatch):
    ds = _make_grid_dataset("radiance")

    def _fake_load(path):
        return ds

    monkeypatch.setattr("alchemi.data.adapters.emit.load_emit_l1b", _fake_load)

    sample = next(iter_emit_pixels("dummy"))
    _assert_geometry(sample)
    _assert_geo(sample)

    radiance_sample_to_toa_reflectance(sample)


def test_enmap_geometry_and_geo(monkeypatch):
    ds = _make_grid_dataset("radiance")
    ds["band_mask"] = ("band", np.ones(3, dtype=bool))

    def _fake_load(path):
        return ds

    monkeypatch.setattr("xarray.load_dataset", lambda path: ds)
    monkeypatch.setattr("alchemi.data.adapters.enmap.load_enmap_l1b", _fake_load)
    monkeypatch.setattr(
        "alchemi.data.adapters.enmap.default_band_widths",
        lambda sensor, wavelengths: np.full_like(wavelengths, 10.0, dtype=float),
    )

    sample = next(iter_enmap_pixels("dummy", srf_blind=True))
    _assert_geometry(sample)
    _assert_geo(sample)

    radiance_sample_to_toa_reflectance(sample)


def test_aviris_ng_geometry_and_geo(monkeypatch):
    ds = _make_grid_dataset("radiance")
    ds["band_mask"] = ("band", np.ones(3, dtype=bool))
    ds["fwhm_nm"] = ("band", np.full(3, 10.0))

    monkeypatch.setattr("alchemi.data.adapters.aviris_ng.load_avirisng_l1b", lambda path: ds)

    sample = next(iter_aviris_ng_pixels("dummy"))
    _assert_geometry(sample)
    _assert_geo(sample)


def test_hytes_geometry_and_geo(monkeypatch):
    ds = _make_grid_dataset("brightness_temp")
    ds["band_mask"] = ("band", np.ones(3, dtype=bool))
    ds.attrs["brightness_temp_units"] = "K"

    monkeypatch.setattr("alchemi.data.adapters.hytes.load_hytes_l1b_bt", lambda path: ds)
    monkeypatch.setattr(
        "alchemi.data.adapters.hytes.default_band_widths",
        lambda sensor, wavelengths: np.full_like(wavelengths, 10.0, dtype=float),
    )

    sample = next(iter_hytes_pixels("dummy", srf_blind=True))
    _assert_geometry(sample)
    _assert_geo(sample)
