import numpy as np
import pytest
import xarray as xr

from alchemi.data.adapters import hytes as hytes_adapter
from alchemi.physics import planck
from alchemi.types import QuantityKind, ValueUnits


pytestmark = pytest.mark.physics_and_metadata


@pytest.fixture()
def synthetic_hytes_dataset():
    wavelengths = np.array([7600.0, 7800.0, 8000.0, 8200.0, 8400.0, 8600.0], dtype=np.float64)
    bt_values_c = np.array(
        [
            [
                [26.0, 26.0, 26.0, 26.0, 26.0, 26.0],
            ]
        ],
        dtype=np.float64,
    )
    ds = xr.Dataset(
        {"brightness_temp": (("y", "x", "band"), bt_values_c)},
        coords={"wavelength_nm": ("band", wavelengths)},
    )
    ds["brightness_temp"].attrs["units"] = "C"
    ds["band_mask"] = ("band", np.array([True, True, False, True, True, True], dtype=bool))
    ds["bad_detector"] = ("band", np.array([False, True, False, False, False, True], dtype=bool))
    return ds


def test_bt_adapter_units_masks_and_provenance(monkeypatch, synthetic_hytes_dataset):
    wavelengths = synthetic_hytes_dataset["wavelength_nm"].values
    widths = np.full_like(wavelengths, 50.0, dtype=np.float64)
    srf_bad_mask = np.array([False, False, False, False, True, False], dtype=bool)
    srf_windows = [(float(wavelengths[3] - 1.0), float(wavelengths[3] + 1.0))]

    def fake_coerce(wavelengths_nm, *, srf_blind):
        dense = hytes_adapter.build_gaussian_srf_matrix(wavelengths_nm, widths, sensor="hytes")
        return dense, "gaussian", srf_bad_mask, srf_windows, widths

    monkeypatch.setattr(hytes_adapter, "_coerce_srf_matrix", fake_coerce)
    monkeypatch.setattr(hytes_adapter, "_load_ds", lambda path: synthetic_hytes_dataset)

    samples = list(hytes_adapter.iter_hytes_pixels("/fake/path.nc"))

    assert len(samples) == 1
    sample = samples[0]

    assert sample.spectrum.kind in {QuantityKind.BRIGHTNESS_T, QuantityKind.BT, "BT", "brightness_temperature"}
    assert sample.spectrum.units == ValueUnits.TEMPERATURE_K
    np.testing.assert_allclose(sample.spectrum.wavelength_nm, wavelengths)

    valid_mask = sample.band_meta.valid_mask
    assert valid_mask.shape == (wavelengths.size,)

    quality_masks = sample.quality_masks
    expected_keys = {"valid_band", "band_mask", "srf_bad_band", "edge_band", "srf_bad_window", "bad_detector"}
    assert expected_keys.issubset(set(quality_masks))

    expected_valid = (
        synthetic_hytes_dataset["band_mask"].values
        & ~srf_bad_mask
        & ~hytes_adapter._edge_mask(wavelengths.size)
        & ~quality_masks["srf_bad_window"]
        & ~synthetic_hytes_dataset["bad_detector"].values
    )
    np.testing.assert_array_equal(quality_masks["valid_band"], expected_valid)

    assert sample.ancillary.get("srf_source") == "gaussian"
    assert sample.band_meta.srf_source[0] == "gaussian"


def test_radiance_path_preserves_metadata(monkeypatch, synthetic_hytes_dataset):
    wavelengths = synthetic_hytes_dataset["wavelength_nm"].values
    widths = np.full_like(wavelengths, 50.0, dtype=np.float64)

    def fake_coerce(wavelengths_nm, *, srf_blind):
        dense = hytes_adapter.build_gaussian_srf_matrix(wavelengths_nm, widths, sensor="hytes")
        return dense, "gaussian", None, None, widths

    monkeypatch.setattr(hytes_adapter, "_coerce_srf_matrix", fake_coerce)
    monkeypatch.setattr(hytes_adapter, "_load_ds", lambda path: synthetic_hytes_dataset)

    bt_sample = next(iter(hytes_adapter.iter_hytes_pixels("/fake/path.nc")))
    rad_sample = next(iter(hytes_adapter.iter_hytes_radiance_pixels("/fake/path.nc")))

    assert rad_sample.spectrum.kind == QuantityKind.RADIANCE
    assert rad_sample.spectrum.units == ValueUnits.RADIANCE_W_M2_SR_NM
    np.testing.assert_allclose(rad_sample.spectrum.wavelength_nm, bt_sample.spectrum.wavelength_nm)
    np.testing.assert_allclose(rad_sample.band_meta.center_nm, bt_sample.band_meta.center_nm)
    np.testing.assert_array_equal(rad_sample.band_meta.valid_mask, bt_sample.band_meta.valid_mask)
    np.testing.assert_equal(rad_sample.quality_masks.keys(), bt_sample.quality_masks.keys())
    for key in rad_sample.quality_masks:
        np.testing.assert_array_equal(rad_sample.quality_masks[key], bt_sample.quality_masks[key])


def test_bt_to_radiance_and_back_roundtrip(monkeypatch, synthetic_hytes_dataset):
    wavelengths = synthetic_hytes_dataset["wavelength_nm"].values
    widths = np.full_like(wavelengths, 50.0, dtype=np.float64)

    def fake_coerce(wavelengths_nm, *, srf_blind):
        dense = hytes_adapter.build_gaussian_srf_matrix(wavelengths_nm, widths, sensor="hytes")
        return dense, "gaussian", None, None, widths

    monkeypatch.setattr(hytes_adapter, "_coerce_srf_matrix", fake_coerce)
    monkeypatch.setattr(hytes_adapter, "_load_ds", lambda path: synthetic_hytes_dataset)

    bt_sample = next(iter(hytes_adapter.iter_hytes_pixels("/fake/path.nc")))
    rad_sample = planck.bt_sample_to_radiance_sample(bt_sample)
    recovered_bt = planck.radiance_sample_to_bt_sample(rad_sample)

    np.testing.assert_allclose(recovered_bt.spectrum.values, bt_sample.spectrum.values, atol=5e-2)
    np.testing.assert_array_equal(recovered_bt.band_meta.valid_mask, bt_sample.band_meta.valid_mask)
    for key in bt_sample.quality_masks:
        np.testing.assert_array_equal(recovered_bt.quality_masks[key], bt_sample.quality_masks[key])
