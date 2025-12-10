import numpy as np
import pytest
import xarray as xr

from alchemi.data.adapters import emit as emit_adapter
from alchemi.types import ValueUnits


pytestmark = pytest.mark.physics_and_metadata


def test_iter_emit_pixels_normalizes_units_and_masks(monkeypatch):
    wavelengths = np.array([1330.0, 1400.0, 1500.0], dtype=np.float64)
    radiance = np.full((1, 1, wavelengths.size), 2.0, dtype=np.float64)
    band_mask = np.array([True, True, False])

    ds = xr.Dataset(
        {
            "radiance": (("y", "x", "band"), radiance),
            "band_mask": (("band",), band_mask),
        },
        coords={"wavelength_nm": ("band", wavelengths)},
    )
    ds["radiance"].attrs["units"] = ValueUnits.RADIANCE_W_M2_SR_UM

    monkeypatch.setattr(emit_adapter, "load_emit_l1b", lambda _path: ds)

    samples = list(emit_adapter.iter_emit_pixels("fake_emit"))
    assert len(samples) == 1

    sample = samples[0]
    assert sample.spectrum.kind == "radiance"
    assert sample.spectrum.units == ValueUnits.RADIANCE_W_M2_SR_NM
    assert np.allclose(sample.spectrum.values, radiance[0, 0, :] * 1e-3)

    assert sample.band_meta is not None
    np.testing.assert_array_equal(sample.band_meta.center_nm, wavelengths)
    assert sample.band_meta.valid_mask.dtype == bool

    expected_valid = np.array([True, False, False])
    np.testing.assert_array_equal(sample.band_meta.valid_mask, expected_valid)
    assert set(sample.quality_masks) >= {"valid_band", "deep_water_vapour", "band_mask"}
    np.testing.assert_array_equal(sample.quality_masks["valid_band"], expected_valid)
    np.testing.assert_array_equal(
        sample.quality_masks["deep_water_vapour"], np.array([False, True, False])
    )


def test_iter_emit_l2a_pixels_scales_and_validates_reflectance(monkeypatch):
    wavelengths = np.array([1330.0, 1400.0, 1500.0], dtype=np.float64)
    reflectance = np.full((1, 1, wavelengths.size), 50.0, dtype=np.float64)
    band_mask = np.array([True, True, False])

    ds = xr.Dataset(
        {
            "reflectance": (("y", "x", "band"), reflectance),
            "band_mask": (("band",), band_mask),
        },
        coords={"wavelength_nm": ("band", wavelengths)},
    )
    ds["reflectance"].attrs["units"] = "%"

    monkeypatch.setattr(emit_adapter, "_load_emit_l2a", lambda _path: ds)

    samples = list(emit_adapter.iter_emit_l2a_pixels("fake_emit"))
    assert len(samples) == 1
    sample = samples[0]
    assert sample.spectrum.kind == "reflectance"
    assert sample.spectrum.units == ValueUnits.REFLECTANCE_FRACTION
    np.testing.assert_allclose(sample.spectrum.values, np.full_like(wavelengths, 0.5))

    expected_valid = np.array([True, False, False])
    np.testing.assert_array_equal(sample.band_meta.valid_mask, expected_valid)
    assert set(sample.quality_masks) >= {"valid_band", "deep_water_vapour", "band_mask"}


def test_iter_emit_l2a_pixels_rejects_out_of_range_reflectance(monkeypatch):
    wavelengths = np.array([500.0], dtype=np.float64)
    reflectance = np.array([[[1.2]]], dtype=np.float64)
    ds = xr.Dataset({"reflectance": (("y", "x", "band"), reflectance)}, coords={"wavelength_nm": ("band", wavelengths)})

    monkeypatch.setattr(emit_adapter, "_load_emit_l2a", lambda _path: ds)

    with pytest.raises(ValueError):
        list(emit_adapter.iter_emit_l2a_pixels("fake_emit"))
