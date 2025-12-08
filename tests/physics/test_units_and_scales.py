import numpy as np
import pytest
import xarray as xr

from alchemi import wavelengths
from alchemi.data.io import emit, enmap
from alchemi.io import mako

pytestmark = pytest.mark.physics_and_metadata


@pytest.mark.parametrize(
    "microns, expected_nm",
    [
        (0.4, 400.0),
        (1.0, 1000.0),
        (2.5, 2500.0),
    ],
)
def test_microns_to_nanometers(microns, expected_nm):
    result = wavelengths.ensure_nm(np.array([microns], dtype=np.float64), "um")
    assert np.allclose(result, np.array([expected_nm], dtype=np.float64))


@pytest.mark.parametrize("nm_values", [np.array([400.0, 1000.0, 2500.0])])
def test_nanometers_round_trip_through_microns(nm_values):
    microns = nm_values / 1000.0
    round_tripped = wavelengths.ensure_nm(microns, "micron")
    assert np.allclose(round_tripped, nm_values)

    unchanged = wavelengths.ensure_nm(nm_values, "nm")
    assert np.allclose(unchanged, nm_values)


@pytest.mark.parametrize(
    "wavenumbers_cm, expected_nm",
    [
        (np.array([1000.0]), np.array([10000.0])),
        (np.array([2000.0]), np.array([5000.0])),
    ],
)
def test_wavenumber_to_nanometer_conversion(wavenumbers_cm, expected_nm):
    converted = wavelengths.to_nm(wavenumbers_cm, "cm^-1")
    assert np.allclose(converted, expected_nm)


def test_wavenumber_increasing_maps_to_decreasing_wavelength():
    wavenumbers_cm = np.array([1000.0, 2000.0, 3000.0])
    converted = wavelengths.to_nm(wavenumbers_cm, "cm-1")
    assert np.all(np.diff(converted) < 0)


@pytest.mark.parametrize(
    "unit_string",
    [
        "W/m^2/sr/um",
        "W m-2 sr-1 um-1",
        "W·m^-2·sr^-1·micron",
        "W/m^2/sr/micrometer",
    ],
)
def test_emit_radiance_scale_round_trip(unit_string):
    base = np.ones(5, dtype=np.float64)
    scale = emit._radiance_scale(unit_string)
    per_nm = base * scale
    restored = per_nm / scale

    assert np.allclose(per_nm, np.full_like(base, scale))
    assert np.allclose(restored, base, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize(
    "unit_string",
    ["W m-2 sr-1 um-1", "mw/m^2/sr/um"],
)
def test_enmap_per_micrometer_scaling(unit_string):
    data = xr.DataArray(np.ones((1, 1, 4)), dims=("y", "x", "band"), attrs={"units": unit_string})
    converted = enmap._convert_radiance(data)

    expected_scale = 1e-6 if "mw" in unit_string.lower() else 1e-3
    assert np.allclose(converted.values, np.ones_like(converted.values) * expected_scale)
    assert converted.attrs["units"] == enmap._RAD_UNITS


def test_mako_microflick_scaling_and_detection():
    base = np.ones(3, dtype=np.float64)
    scaled = base * mako._MICROFLICK_TO_W_M2_SR_NM

    assert mako._is_microflick("microflick") is True
    assert mako._is_microflick("µW/cm^2/sr/µm") is True
    assert np.allclose(scaled, np.full_like(base, mako._MICROFLICK_TO_W_M2_SR_NM))


@pytest.mark.parametrize("units", ["feet", "kHz"])
def test_invalid_wavelength_units_raise(units):
    with pytest.raises(ValueError):
        wavelengths.to_nm(np.array([1.0, 2.0]), units)


@pytest.mark.parametrize("units", [None, "unknown-unit"])
def test_invalid_or_missing_radiance_units_scale_to_one(units):
    assert emit._radiance_scale(units) == 1.0
