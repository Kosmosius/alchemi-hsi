import importlib
import types

import numpy as np
import xarray as xr

import alchemi.data.io as data_io


def _stub_dataset(wavelengths: np.ndarray, values: np.ndarray, key: str) -> xr.Dataset:
    return xr.Dataset({key: (("y", "x", "band"), values)}, coords={"wavelength_nm": ("band", wavelengths)})


def test_enmap_adapter_preserves_masks_and_units(tmp_path, monkeypatch):
    wavelengths = np.array([400.0, 500.0])
    radiance = np.ones((1, 1, wavelengths.size), dtype=np.float64)
    band_mask = np.array([True, False])
    fwhm = np.array([10.0, 10.0])
    ds = xr.Dataset(
        {
            "radiance": (("y", "x", "band"), radiance),
            "band_mask": (("band",), band_mask),
            "fwhm_nm": (("band",), fwhm),
        },
        coords={"wavelength_nm": ("band", wavelengths)},
    )
    path = tmp_path / "enmap.nc"
    ds.to_netcdf(path)

    monkeypatch.setattr(data_io, "load_enmap_l1b", lambda *_args, **_kwargs: ds)
    enmap = importlib.import_module("alchemi.data.adapters.enmap")
    monkeypatch.setattr(enmap.srfs, "get_srf", lambda *_args, **_kwargs: None)

    samples = list(enmap.iter_enmap_pixels(path))
    assert len(samples) == 1
    sample = samples[0]
    assert sample.spectrum.kind == "radiance"
    assert sample.band_meta is not None
    assert bool(sample.band_meta.valid_mask[1]) is False
    assert "band_mask" in sample.quality_masks


def test_hytes_adapter_emits_bt_samples(tmp_path, monkeypatch):
    wavelengths = np.array([8000.0, 8100.0])
    bt = np.array([[[290.0, 295.0]]])
    ds = _stub_dataset(wavelengths, bt, "brightness_temperature")
    path = tmp_path / "hytes.nc"
    ds.to_netcdf(path)

    monkeypatch.setattr(data_io, "load_hytes_l1b", lambda _path: xr.open_dataset(_path))
    hytes_module = importlib.import_module("alchemi.data.adapters.hytes")
    monkeypatch.setattr(hytes_module.srfs, "get_srf", lambda *_args, **_kwargs: None)
    samples = list(hytes_module.iter_hytes_pixels(path))
    assert len(samples) == 1
    sample = samples[0]
    assert sample.spectrum.kind == "BT"
    assert str(sample.ancillary["source_path"]) == str(path)
