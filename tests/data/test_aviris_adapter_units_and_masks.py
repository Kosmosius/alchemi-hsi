import numpy as np
import pytest
import xarray as xr

from alchemi.data.adapters import aviris_ng as aviris_adapter


pytestmark = pytest.mark.physics_and_metadata


def test_iter_aviris_ng_pixels_resolves_wavelengths_and_masks(monkeypatch):
    wavelengths_um = np.array([1.3, 1.4, 1.6], dtype=np.float64)
    expected_nm = wavelengths_um * 1000.0
    radiance = np.ones((1, 1, wavelengths_um.size), dtype=np.float64)
    band_mask = np.array([True, True, False])

    fwhm = np.array([8.0, 8.0, 8.0], dtype=np.float64)

    ds = xr.Dataset(
        {
            "radiance": (("y", "x", "band"), radiance),
            "band_mask": (("band",), band_mask),
            "fwhm_nm": (("band",), fwhm),
        },
        coords={"wavelength": ("band", wavelengths_um)},
    )

    monkeypatch.setattr(aviris_adapter, "load_avirisng_l1b", lambda _path: ds)

    samples = list(aviris_adapter.iter_aviris_ng_pixels("fake_aviris", srf_blind=True))
    assert len(samples) == 1

    sample = samples[0]
    np.testing.assert_array_equal(sample.spectrum.wavelength_nm, expected_nm)
    assert sample.spectrum.kind == "radiance"

    expected_deep_water = np.array([False, True, False])
    expected_valid = np.array([True, False, False])
    np.testing.assert_array_equal(sample.band_meta.center_nm, expected_nm)
    np.testing.assert_array_equal(sample.band_meta.valid_mask, expected_valid)

    assert "deep_water_vapour" in sample.quality_masks
    np.testing.assert_array_equal(sample.quality_masks["deep_water_vapour"], expected_deep_water)
    np.testing.assert_array_equal(sample.quality_masks["valid_band"], expected_valid)
