import numpy as np
import xarray as xr

from alchemi.data.adapters import emit as emit_adapter
from alchemi.types import SRFMatrix


def _stub_srf():
    centers = np.array([400.0, 500.0, 600.0])
    nm = [np.linspace(395, 405, 5), np.linspace(495, 505, 5), np.linspace(595, 605, 5)]
    resp = [np.ones(5), np.ones(5), np.ones(5)]
    return SRFMatrix(sensor="emit", centers_nm=centers, bands_nm=nm, bands_resp=resp)


def test_iter_emit_pixels_normalises_radiance_and_masks(monkeypatch, tmp_path):
    wavelengths = np.array([400.0, 500.0, 600.0])
    radiance = np.ones((1, 1, wavelengths.size), dtype=np.float64)
    band_mask = np.array([True, False, True])
    ds = xr.Dataset(
        {
            "radiance": (("y", "x", "band"), radiance, {"units": "W/m^2/sr/um"}),
            "band_mask": (("band",), band_mask),
        },
        coords={"wavelength_nm": ("band", wavelengths)},
    )

    monkeypatch.setattr(emit_adapter, "load_emit_l1b", lambda *_args, **_kwargs: ds)
    monkeypatch.setattr(emit_adapter.srfs, "get_srf", lambda *_args, **_kwargs: _stub_srf())

    samples = list(emit_adapter.iter_emit_pixels(str(tmp_path / "emit")))
    assert len(samples) == 1
    sample = samples[0]
    assert sample.spectrum.kind == "radiance"
    assert np.isclose(sample.spectrum.values[0], 0.001)  # converted from per-µm to per-nm
    assert "valid_band" in sample.quality_masks
    np.testing.assert_array_equal(sample.quality_masks["valid_band"], band_mask)
    assert sample.ancillary["srf_source"] == "official"
    assert sample.srf_matrix is not None


def test_iter_emit_l2a_pixels_reflectance(monkeypatch, tmp_path):
    wavelengths = np.array([400.0, 500.0])
    reflectance = np.array([[[10.0, 20.0]]])
    ds = xr.Dataset(
        {
            "reflectance": (("y", "x", "band"), reflectance, {"units": "percent"}),
            "band_mask": (("band",), np.array([True, True])),
        },
        coords={"wavelength_nm": ("band", wavelengths, {"units": "nm"})},
    )
    path = tmp_path / "emit_l2a.nc"
    ds.to_netcdf(path)

    monkeypatch.setattr(emit_adapter.srfs, "get_srf", lambda *_args, **_kwargs: _stub_srf())

    samples = list(emit_adapter.iter_emit_l2a_pixels(path))
    assert len(samples) == 1
    sample = samples[0]
    assert sample.spectrum.kind == "reflectance"
    np.testing.assert_allclose(sample.spectrum.values, np.array([0.1, 0.2]))
    np.testing.assert_array_equal(sample.quality_masks["valid_band"], np.array([True, True]))


def test_attach_emit_l2b_labels(monkeypatch, tmp_path):
    wavelengths = np.array([400.0, 500.0])
    radiance = np.ones((1, 1, wavelengths.size))
    ds = xr.Dataset({"radiance": (("y", "x", "band"), radiance)}, coords={"wavelength_nm": ("band", wavelengths)})
    monkeypatch.setattr(emit_adapter, "load_emit_l1b", lambda *_args, **_kwargs: ds)
    monkeypatch.setattr(emit_adapter.srfs, "get_srf", lambda *_args, **_kwargs: None)

    samples = list(emit_adapter.iter_emit_pixels("dummy"))

    mineral = xr.DataArray([["ILLITE_MUSCOVITE_GROUP"]], dims=("y", "x"))
    r2 = xr.DataArray([[0.95]], dims=("y", "x"))
    ds_l2b = xr.Dataset({"mineral_group": mineral, "fit_r2": r2})

    updated = emit_adapter.attach_emit_l2b_labels(samples, ds_l2b)
    assert updated[0].ancillary["labels"]["emit_l2b"] == {
        "mineral_group": "ILLITE_MUSCOVITE_GROUP",
        "fit_r2": 0.95,
    }


def test_emit_srf_blind_gaussian(monkeypatch, tmp_path):
    wavelengths = np.array([400.0, 500.0, 600.0])
    radiance = np.ones((1, 1, wavelengths.size), dtype=np.float64)
    band_mask = np.array([True, True, True])
    ds = xr.Dataset(
        {"radiance": (("y", "x", "band"), radiance, {"units": "W/m^2/sr/nm"}), "band_mask": (("band",), band_mask)},
        coords={"wavelength_nm": ("band", wavelengths)},
    )

    monkeypatch.setattr(emit_adapter, "load_emit_l1b", lambda *_args, **_kwargs: ds)
    monkeypatch.setattr(emit_adapter.srfs, "get_srf", lambda *_args, **_kwargs: _stub_srf())

    samples = list(emit_adapter.iter_emit_pixels("dummy", srf_blind=True))
    assert len(samples) == 1
    sample = samples[0]
    assert sample.ancillary.get("srf_mode") == "srf-blind"
    assert sample.band_meta is not None
    assert sample.band_meta.srf_source[0] == "gaussian"
    assert sample.srf_matrix is not None
