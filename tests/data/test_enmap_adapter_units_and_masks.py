import numpy as np
import pytest
import xarray as xr

from alchemi.data.adapters import enmap as enmap_adapter
from alchemi.types import ValueUnits


pytestmark = pytest.mark.physics_and_metadata


def test_iter_enmap_pixels_normalizes_units_and_water_vapour_mask(monkeypatch):
    wavelengths = np.array([1320.0, 1400.0, 1500.0], dtype=np.float64)
    radiance = np.full((1, 1, wavelengths.size), 2.0, dtype=np.float64)
    band_mask = np.array([True, True, False])

    fwhm = np.array([10.0, 10.0, 10.0], dtype=np.float64)

    ds = xr.Dataset(
        {
            "radiance": (("y", "x", "band"), radiance),
            "band_mask": (("band",), band_mask),
            "fwhm_nm": (("band",), fwhm),
        },
        coords={"wavelength_nm": ("band", wavelengths)},
    )
    ds["radiance"].attrs["units"] = "mW m-2 sr-1 um-1"

    monkeypatch.setattr(enmap_adapter, "_normalize_l1_dataset", lambda _path: ds)
    monkeypatch.setattr(enmap_adapter.srfs, "get_srf", lambda *_args, **_kwargs: None)

    samples = list(enmap_adapter.iter_enmap_pixels("fake_enmap"))
    assert len(samples) == 1

    sample = samples[0]
    assert sample.spectrum.units == ValueUnits.RADIANCE_W_M2_SR_NM
    np.testing.assert_allclose(sample.spectrum.values, np.full_like(wavelengths, 2e-6))

    expected_valid = np.array([True, False, False])
    np.testing.assert_array_equal(sample.band_meta.center_nm, wavelengths)
    np.testing.assert_array_equal(sample.band_meta.valid_mask, expected_valid)

    assert set(sample.quality_masks) >= {"valid_band", "deep_water_vapour", "band_mask"}
    np.testing.assert_array_equal(
        sample.quality_masks["deep_water_vapour"], np.array([False, True, False])
    )
    np.testing.assert_array_equal(sample.quality_masks["valid_band"], expected_valid)
