import numpy as np
import pytest
import xarray as xr

from alchemi.data.io.avirisng import avirisng_pixel
from alchemi.data.io.emit import emit_pixel
from alchemi.data.io.enmap import enmap_pixel
from alchemi.data.io.hytes import hytes_pixel_bt
from alchemi.types import (
    QuantityKind,
    RadianceUnits,
    ReflectanceUnits,
    Spectrum,
    TemperatureUnits,
    ValueUnits,
)


def _toy_ds(
    var_name: str, values: np.ndarray, wavelengths: np.ndarray, *, units: str
) -> xr.Dataset:
    ds = xr.Dataset(
        {var_name: (("y", "x", "band"), values, {"units": units})},
        coords={
            "y": np.arange(values.shape[0]),
            "x": np.arange(values.shape[1]),
            "band": np.arange(values.shape[2]),
            "wavelength_nm": ("band", wavelengths),
        },
    )
    return ds


def test_strict_monotonicity_and_shapes():
    with pytest.raises(ValueError):
        Spectrum(wavelengths=[500.0, 400.0], values=np.ones((2,)), kind=QuantityKind.RADIANCE)
    with pytest.raises(ValueError):
        Spectrum(wavelengths=[400.0, 400.0], values=np.ones((2,)), kind=QuantityKind.RADIANCE)


def test_broadcast_and_default_units():
    values = np.ones((2, 2, 3))
    spec = Spectrum(wavelengths=[1.0, 2.0, 3.0], values=values, kind=QuantityKind.REFLECTANCE)
    assert spec.values.shape == values.shape
    assert spec.flatten_spatial().shape == (4, 3)
    assert spec.units == ValueUnits.REFLECTANCE_FRACTION


def test_emit_pixel_radiance_units():
    wavelengths = np.array([400.0, 500.0, 600.0])
    radiance = np.ones((1, 1, 3)) * 5.0
    ds = _toy_ds("radiance", radiance, wavelengths, units=RadianceUnits.W_M2_SR_NM.value)
    ds.attrs["sensor"] = "EMIT"
    spec = emit_pixel(ds, 0, 0)
    assert spec.kind == QuantityKind.RADIANCE
    assert spec.units == ValueUnits.RADIANCE_W_M2_SR_NM
    np.testing.assert_allclose(spec.values, radiance[0, 0, :])


def test_enmap_pixel_radiance_units():
    wavelengths = np.array([400.0, 500.0])
    radiance = np.full((1, 1, 2), 2.5)
    ds = _toy_ds("radiance", radiance, wavelengths, units=RadianceUnits.W_M2_SR_NM.value)
    ds.attrs.update(sensor="enmap", units=RadianceUnits.W_M2_SR_NM)
    spec = enmap_pixel(ds, 0, 0)
    assert spec.kind == QuantityKind.RADIANCE
    assert spec.units == ValueUnits.RADIANCE_W_M2_SR_NM


def test_aviris_pixel_radiance_units():
    wavelengths = np.array([1000.0, 1100.0, 1200.0])
    radiance = np.full((1, 1, 3), 1.2)
    ds = _toy_ds("radiance", radiance, wavelengths, units=RadianceUnits.W_M2_SR_NM.value)
    ds.attrs["sensor"] = "avirisng"
    spec = avirisng_pixel(ds, 0, 0)
    assert spec.kind == QuantityKind.RADIANCE
    assert spec.units == ValueUnits.RADIANCE_W_M2_SR_NM


def test_hytes_pixel_bt_units():
    wavelengths = np.array([8000.0, 9000.0])
    bt = np.full((1, 1, 2), 300.0)
    ds = _toy_ds("brightness_temp", bt, wavelengths, units="K")
    ds.attrs["sensor"] = "HyTES"
    spec = hytes_pixel_bt(ds, 0, 0)
    assert spec.kind == QuantityKind.BRIGHTNESS_T
    assert spec.units == ValueUnits.TEMPERATURE_K


def test_reflectance_units_wire():
    wavelengths = np.array([400.0, 500.0])
    spec = Spectrum(
        wavelengths=wavelengths,
        values=np.array([0.1, 0.2]),
        kind=QuantityKind.REFLECTANCE,
        units=ReflectanceUnits.FRACTION,
    )
    assert spec.units == ValueUnits.REFLECTANCE_FRACTION
