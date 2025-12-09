import numpy as np
import pytest
import xarray as xr

from alchemi.data.adapters import hytes as hytes_adapter
from alchemi.data.io.hytes import HYTES_BAND_COUNT, HYTES_WAVELENGTHS_NM
from alchemi.physics import planck
from alchemi.physics.planck import planck_radiance_wavelength, radiance_to_bt_K


pytestmark = pytest.mark.physics_and_metadata


def _write_hytes_bt_cube(tmp_path, value: float = 300.0) -> str:
    bt = np.full((1, 1, HYTES_BAND_COUNT), value, dtype=np.float64)
    ds = xr.Dataset(
        {"brightness_temp": (("y", "x", "band"), bt)},
        coords={"wavelength_nm": ("band", HYTES_WAVELENGTHS_NM)},
    )
    path = tmp_path / "hytes_bt.nc"
    ds.to_netcdf(path)
    return str(path)


def test_hytes_bt_sample_masks_and_metadata(tmp_path):
    path = _write_hytes_bt_cube(tmp_path, value=295.0)
    samples = list(hytes_adapter.iter_hytes_pixels(path))
    assert len(samples) == 1
    sample = samples[0]

    assert sample.spectrum.kind.value == "brightness_temperature"
    assert np.allclose(sample.spectrum.values[0], 295.0)

    valid = sample.band_meta.valid_mask
    assert valid.shape[0] == HYTES_BAND_COUNT
    # Edge bands should be excluded from the valid mask.
    assert not valid[0]
    assert not valid[-1]

    assert sample.quality_masks["valid_band"].shape[0] == HYTES_BAND_COUNT
    assert sample.ancillary["source_path"] == path
    assert sample.ancillary["srf_source"] in {"official", "gaussian", "none"}
    assert sample.ancillary.get("srf_mode") == "srf-aware"


def test_hytes_radiance_conversion_round_trip(tmp_path):
    path = _write_hytes_bt_cube(tmp_path, value=310.0)

    bt_sample = next(iter(hytes_adapter.iter_hytes_pixels(path)))
    rad_sample = next(iter(hytes_adapter.iter_hytes_radiance_pixels(path)))

    expected_radiance = planck_radiance_wavelength(
        bt_sample.spectrum.wavelength_nm, bt_sample.spectrum.values
    )
    if bt_sample.srf_matrix is not None:
        expected_radiance = planck.bt_spectrum_to_radiance(
            bt_sample.spectrum,
            srf_matrix=bt_sample.srf_matrix.matrix,
            srf_wavelength_nm=bt_sample.srf_matrix.wavelength_nm,
        ).values
    assert rad_sample.spectrum.kind.value == "radiance"
    assert np.allclose(rad_sample.spectrum.values, expected_radiance, atol=1e-6, rtol=1e-4)

    recovered_bt = radiance_to_bt_K(rad_sample.spectrum.values, rad_sample.spectrum.wavelength_nm)
    assert np.allclose(recovered_bt, bt_sample.spectrum.values, atol=5e-2)

    assert np.array_equal(
        bt_sample.quality_masks["valid_band"], rad_sample.quality_masks["valid_band"]
    )


def test_hytes_srf_blind_gaussian(tmp_path):
    path = _write_hytes_bt_cube(tmp_path, value=300.0)
    samples = list(hytes_adapter.iter_hytes_pixels(path, srf_blind=True))
    assert len(samples) == 1
    sample = samples[0]
    assert sample.ancillary.get("srf_mode") == "srf-blind"
    assert sample.band_meta is not None
    assert sample.band_meta.srf_source[0] == "gaussian"
    assert sample.srf_matrix is not None
