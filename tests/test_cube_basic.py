from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from alchemi.data.cube import Cube, GeoInfo, geo_from_attrs
from alchemi.types import QuantityKind, RadianceUnits, TemperatureUnits
from alchemi.ingest import (
    from_avirisng_l1b,
    from_emit_l1b,
    from_enmap_l1b,
    from_hytes_bt,
    from_mako_l2s,
    from_mako_l3,
)


def _make_dataset(var_name: str, *, units: str, sensor: str) -> xr.Dataset:
    y, x, band = 2, 3, 4
    values = np.arange(y * x * band, dtype=np.float64).reshape(y, x, band)
    coords = {
        "y": np.arange(y, dtype=np.int64),
        "x": np.arange(x, dtype=np.int64),
        "band": np.arange(band, dtype=np.int64),
    }

    ds = xr.Dataset()
    ds[var_name] = xr.DataArray(
        values,
        dims=("y", "x", "band"),
        coords=coords,
        attrs={"units": units},
    )
    ds = ds.assign_coords(wavelength_nm=("band", np.linspace(400.0, 800.0, band)))
    ds.attrs.update(
        sensor=sensor,
        crs="EPSG:4326",
        transform=(0.0, 30.0, 15.0, 45.0, 0.0, -30.0),
    )
    return ds


def test_cube_validation_shape() -> None:
    data = np.zeros((4, 5), dtype=np.float32)
    with pytest.raises(ValueError):
        Cube(
            data=data,
            axis=np.linspace(0.0, 1.0, 5),
            axis_unit="wavelength_nm",
            value_kind="radiance",
        )


def test_cube_validation_axis_length() -> None:
    data = np.zeros((4, 5, 6), dtype=np.float32)
    with pytest.raises(ValueError):
        Cube(
            data=data,
            axis=np.linspace(0.0, 1.0, 5),
            axis_unit="wavelength_nm",
            value_kind="radiance",
        )


def test_cube_validation_axis_unit() -> None:
    data = np.zeros((2, 2, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        Cube(
            data=data,
            axis=np.linspace(0.0, 1.0, 3),
            axis_unit="frequency",
            value_kind="radiance",
        )


def test_cube_validation_value_kind() -> None:
    data = np.zeros((2, 2, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        Cube(
            data=data,
            axis=np.linspace(0.0, 1.0, 3),
            axis_unit="wavelength_nm",
            value_kind="invalid",
        )


def test_cube_accepts_geo_and_attrs() -> None:
    data = np.zeros((1, 1, 1), dtype=np.float64)
    cube = Cube(
        data=data,
        axis=np.array([1.0]),
        axis_unit="wavenumber_cm1",
        value_kind="reflectance",
        geo=GeoInfo(crs="epsg:4326", transform=(0, 1, 2, 3, 4, 5)),
        attrs={"sensor": "test"},
    )
    assert cube.geo is not None
    assert cube.geo.crs == "epsg:4326"
    assert cube.attrs["sensor"] == "test"


@pytest.mark.parametrize(
    "adapter, var_name, units, value_kind, sensor",
    [
        (from_emit_l1b, "radiance", RadianceUnits.W_M2_SR_NM.value, QuantityKind.RADIANCE, "emit"),
        (from_enmap_l1b, "radiance", RadianceUnits.W_M2_SR_NM.value, QuantityKind.RADIANCE, "enmap"),
        (from_avirisng_l1b, "radiance", RadianceUnits.W_M2_SR_NM.value, QuantityKind.RADIANCE, "avirisng"),
        (from_hytes_bt, "brightness_temp", TemperatureUnits.KELVIN.value, QuantityKind.BRIGHTNESS_T, "hytes"),
        (from_mako_l2s, "radiance", RadianceUnits.W_M2_SR_NM.value, QuantityKind.RADIANCE, "mako"),
        (from_mako_l3, "bt", TemperatureUnits.KELVIN.value, QuantityKind.BRIGHTNESS_T, "mako"),
    ],
)
def test_adapter_round_trip(
    adapter, var_name: str, units: str, value_kind: QuantityKind, sensor: str
) -> None:
    ds = _make_dataset(var_name, units=units, sensor=sensor)
    cube = adapter(ds)

    np.testing.assert_allclose(cube.data, ds[var_name].values)
    np.testing.assert_allclose(cube.axis, ds["wavelength_nm"].values)
    assert cube.axis_unit == "wavelength_nm"
    assert cube.value_kind is value_kind
    assert cube.attrs["sensor"] == sensor
    assert cube.geo is not None
    assert cube.geo.crs == "EPSG:4326"


def test_geo_from_attrs() -> None:
    attrs = {"crs": "EPSG:4326", "transform": (1, 0, 0, 0, 1, 0)}
    geo = geo_from_attrs(attrs)
    assert isinstance(geo, GeoInfo)
    assert geo.crs == "EPSG:4326"
    assert geo.transform == (1, 0, 0, 0, 1, 0)
