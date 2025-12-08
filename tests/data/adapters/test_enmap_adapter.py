import numpy as np
import pytest
import xarray as xr

from alchemi.data.adapters.enmap import (
    iter_enmap_l2a_pixels,
    iter_enmap_pixels,
    load_enmap_l2a_scene,
    load_enmap_scene,
)
from alchemi.registry import srfs


pytestmark = pytest.mark.physics_and_metadata


def _write_enmap_cube(
    path, wavelengths_nm, *, var_name="radiance", units="W m-2 sr-1 nm-1", mask=None
):
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=np.float64)
    band_count = wavelengths_nm.shape[0]
    data = np.full((1, 1, band_count), 1.0, dtype=np.float64)
    fwhm = np.full(band_count, 12.0, dtype=np.float64)

    ds = xr.Dataset(
        {var_name: (("y", "x", "band"), data, {"units": units})},
        coords={"wavelength_nm": ("band", wavelengths_nm)},
    )
    ds["fwhm_nm"] = ("band", fwhm)
    if mask is not None:
        ds["band_mask"] = ("band", np.asarray(mask, dtype=bool))
    ds.to_netcdf(path)


def test_enmap_l1b_samples_include_srf_and_masks(tmp_path):
    srf = srfs.get_srf("enmap")
    wavelengths = srf.centers_nm
    band_mask = np.ones_like(wavelengths, dtype=bool)
    band_mask[0] = False
    cube_path = tmp_path / "l1c.nc"
    _write_enmap_cube(cube_path, wavelengths, mask=band_mask)

    samples = load_enmap_scene(cube_path)
    assert len(samples) == 1
    sample = samples[0]

    assert np.all(np.diff(sample.spectrum.wavelength_nm) > 0)
    assert sample.spectrum.kind == "radiance"
    assert sample.srf_matrix is not None
    row_area = np.trapezoid(sample.srf_matrix.matrix, x=sample.srf_matrix.wavelength_nm, axis=1)
    np.testing.assert_allclose(row_area, np.ones_like(row_area))
    assert sample.ancillary.get("srf_mode") == "srf-aware"

    valid = sample.quality_masks["valid_band"]
    assert valid.shape[0] == wavelengths.shape[0]
    assert not valid[0]
    assert np.any(~valid)


def test_enmap_l2a_reflectance_clipped_and_masked(tmp_path):
    wavelengths = np.array([430.0, 900.0, 1400.0], dtype=np.float64)
    mask = np.array([True, False, True])
    cube_path = tmp_path / "l2a.nc"
    _write_enmap_cube(
        cube_path,
        wavelengths,
        var_name="reflectance",
        units="percent",
        mask=mask,
    )

    samples = load_enmap_l2a_scene(cube_path)
    sample = samples[0]

    assert sample.spectrum.kind == "reflectance"
    assert np.all((sample.spectrum.values >= 0) & (sample.spectrum.values <= 1))
    valid = sample.quality_masks["valid_band"]
    assert not valid[1]  # band mask invalid
    assert not valid[2]  # absorption window


def test_iterators_stream_pixels(tmp_path):
    wavelengths = np.array([430.0, 900.0], dtype=np.float64)
    cube_path = tmp_path / "l1c.nc"
    refl_path = tmp_path / "l2a.nc"
    _write_enmap_cube(cube_path, wavelengths)
    _write_enmap_cube(refl_path, wavelengths, var_name="reflectance")

    samples = list(iter_enmap_pixels(cube_path))
    assert len(samples) == 1

    l2a_samples = list(iter_enmap_l2a_pixels(refl_path))
    assert len(l2a_samples) == 1
    assert l2a_samples[0].spectrum.kind == "reflectance"


def test_enmap_srf_blind_uses_gaussian(tmp_path, monkeypatch):
    wavelengths = np.array([430.0, 900.0], dtype=np.float64)
    cube_path = tmp_path / "l1c.nc"
    _write_enmap_cube(cube_path, wavelengths)

    monkeypatch.setattr(srfs, "get_srf", lambda *_args, **_kwargs: srfs.get_srf("enmap"))

    samples = list(iter_enmap_pixels(cube_path, srf_blind=True))
    assert len(samples) == 1
    sample = samples[0]
    assert sample.ancillary.get("srf_mode") == "srf-blind"
    assert sample.band_meta is not None
    assert sample.band_meta.srf_source[0] == "gaussian"
    assert sample.srf_matrix is not None
