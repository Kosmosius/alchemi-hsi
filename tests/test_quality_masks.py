import numpy as np
import xarray as xr

from alchemi.data.adapters import aviris_ng, emit, enmap, hytes
from alchemi.data.sample_utils import compute_usable_mask
from alchemi.spectral import Sample, Spectrum


def test_compute_usable_mask_respects_deep_water():
    wavelengths = np.array([1000.0, 1400.0, 1500.0], dtype=float)
    sample = Sample(
        spectrum=Spectrum(wavelength_nm=wavelengths, values=np.zeros_like(wavelengths), kind="radiance"),
        sensor_id="dummy",
        quality_masks={
            "valid_band": np.array([True, True, False], dtype=bool),
            "deep_water_vapour": np.array([False, True, False], dtype=bool),
        },
    )

    usable_default = compute_usable_mask(sample)
    np.testing.assert_array_equal(usable_default, np.array([True, False, False]))

    usable_all = compute_usable_mask(sample, for_task="all")
    np.testing.assert_array_equal(usable_all, np.array([True, True, False]))


def test_emit_quality_masks_include_valid_band_and_water_mask():
    wavelengths = np.array([1000.0, 1400.0, 1600.0, 1900.0], dtype=float)
    coords = {"y": [0], "x": [0], "band": np.arange(wavelengths.size)}
    radiance = xr.DataArray(
        np.zeros((1, 1, wavelengths.size), dtype=float),
        dims=("y", "x", "band"),
        coords={**coords, "wavelength_nm": ("band", wavelengths)},
    )
    ds = xr.Dataset({"radiance": radiance, "band_mask": ("band", np.array([True, True, False, True], dtype=bool))})
    ds = ds.assign_coords({"wavelength_nm": ("band", wavelengths)})

    quality = emit._quality_masks(ds, radiance, include_quality=True)
    valid_band = quality["valid_band"][0, 0, :]
    deep_water = quality.get("deep_water_vapour")

    np.testing.assert_array_equal(valid_band, np.array([True, False, False, False]))
    assert deep_water is not None
    np.testing.assert_array_equal(deep_water[0, 0, :], np.array([False, True, False, True]))


def test_enmap_valid_band_filters_absorption_windows():
    wavelengths = np.array([1200.0, 1350.0, 1500.0, 1850.0], dtype=float)
    dataset_mask = np.array([True, True, True, True], dtype=bool)

    valid, deep_water_vapour = enmap._valid_band_mask(wavelengths, dataset_mask=dataset_mask, srf=None)

    assert deep_water_vapour is not None
    np.testing.assert_array_equal(deep_water_vapour, np.array([False, True, False, True]))
    np.testing.assert_array_equal(valid, np.array([True, False, True, False]))


def test_aviris_quality_masks_expose_deep_water():
    wavelengths = np.array([1000.0, 1400.0, 2000.0], dtype=float)
    band_mask = np.array([True, True, True], dtype=bool)
    deep_mask = aviris_ng._deep_water_vapour_mask(wavelengths)
    quality = aviris_ng._quality_masks(band_mask, deep_water_vapour=deep_mask)

    np.testing.assert_array_equal(quality["deep_water_vapour"], np.array([False, True, False]))
    np.testing.assert_array_equal(quality["valid_band"], np.array([True, False, True]))


def test_hytes_valid_band_respects_detector_and_edges():
    wavelengths = np.array([7600.0, 7700.0, 7800.0, 7900.0, 8000.0], dtype=float)
    dataset_mask = np.array([True, True, True, True, True], dtype=bool)
    detector_mask = np.array([False, True, False, False, False], dtype=bool)

    valid, quality = hytes._valid_band_mask(
        wavelengths, dataset_mask=dataset_mask, srf_mask=None, srf_windows=[], detector_mask=detector_mask
    )

    np.testing.assert_array_equal(quality["bad_detector"], detector_mask)
    np.testing.assert_array_equal(valid, np.array([False, False, True, False, False]))
