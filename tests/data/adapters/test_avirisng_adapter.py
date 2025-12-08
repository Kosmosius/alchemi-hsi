import numpy as np
import pytest
import xarray as xr

from alchemi.data.adapters import aviris_ng as aviris_adapter

pytestmark = pytest.mark.physics_and_metadata


def _mock_cube(wavelengths: np.ndarray) -> xr.Dataset:
    radiance = np.ones((1, 1, wavelengths.size), dtype=np.float64)
    fwhm = np.full_like(wavelengths, 10.0, dtype=np.float64)
    return xr.Dataset(
        {
            "radiance": (("y", "x", "band"), radiance),
            "fwhm_nm": (("band",), fwhm),
            "band_mask": (("band",), np.ones_like(wavelengths, dtype=bool)),
        },
        coords={"wavelength_nm": ("band", wavelengths)},
    )


def test_avirisng_srf_modes(monkeypatch, tmp_path):
    wavelengths = np.array([1_000.0, 1_020.0, 1_040.0], dtype=np.float64)
    ds = _mock_cube(wavelengths)
    monkeypatch.setattr(aviris_adapter, "load_avirisng_l1b", lambda *_args, **_kwargs: ds)
    monkeypatch.setattr(aviris_adapter.srfs, "get_srf", lambda *_args, **_kwargs: None)

    aware = list(aviris_adapter.iter_aviris_ng_pixels("path", srf_blind=False))
    blind = list(aviris_adapter.iter_aviris_ng_pixels("path", srf_blind=True))

    assert aware[0].ancillary.get("srf_mode") == "srf-aware"
    assert blind[0].ancillary.get("srf_mode") == "srf-blind"
    assert blind[0].band_meta is not None
    assert blind[0].band_meta.srf_source[0] == "gaussian"
    assert blind[0].srf_matrix is not None
