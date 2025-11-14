import numpy as np
import pytest
import xarray as xr

from alchemi.physics import bt_to_radiance, radiance_to_bt
from alchemi.types import SpectrumKind
from alchemi.data.io import (
    HYTES_BAND_COUNT,
    HYTES_WAVELENGTHS_NM,
    hytes_pixel_bt,
    load_hytes_l1b_bt,
)


def test_hytes_wavelengths_ok():
    assert HYTES_WAVELENGTHS_NM.shape == (HYTES_BAND_COUNT,)
    assert np.isclose(HYTES_WAVELENGTHS_NM[0], 7_500.0)
    assert np.isclose(HYTES_WAVELENGTHS_NM[-1], 12_000.0)
    diffs = np.diff(HYTES_WAVELENGTHS_NM)
    assert np.all(diffs > 0)


def test_load_hytes_l1b_bt_and_pixel(tmp_path):
    rng = np.random.default_rng(42)
    data = rng.uniform(250.0, 320.0, size=(HYTES_BAND_COUNT, 4, 3))

    raw = xr.Dataset(
        {
            "BrightnessTemperature": (("band", "y", "x"), data),
        },
        coords={
            "band": np.arange(HYTES_BAND_COUNT, dtype=np.int32),
            "y": np.arange(4, dtype=np.int32),
            "x": np.arange(3, dtype=np.int32),
        },
    )
    raw["BrightnessTemperature"].attrs["units"] = "K"
    raw.attrs["bt_units"] = "K"

    path = tmp_path / "hytes_bt.nc"
    raw.to_netcdf(path)

    loaded = load_hytes_l1b_bt(str(path))

    assert set(loaded.sizes.keys()) == {"y", "x", "band"}
    bt = loaded["brightness_temp"]
    assert bt.values.shape == (4, 3, HYTES_BAND_COUNT)
    assert bt.dims == ("y", "x", "band")
    assert bt.attrs["units"] == "K"
    band_mask = loaded["band_mask"]
    assert band_mask.values.dtype == bool
    assert np.all(band_mask.values)
    assert "wavelength_nm" in loaded.coords
    assert loaded.coords["wavelength_nm"].attrs["units"] == "nm"
    np.testing.assert_allclose(loaded["wavelength_nm"].values, HYTES_WAVELENGTHS_NM)

    spec = hytes_pixel_bt(loaded, 1, 2)
    assert spec.kind is SpectrumKind.BT
    assert spec.units == "K"
    np.testing.assert_allclose(spec.values, bt.sel(y=1, x=2).values)
    np.testing.assert_allclose(spec.wavelengths.nm, HYTES_WAVELENGTHS_NM)
    assert spec.mask is not None
    assert spec.mask.shape == (HYTES_BAND_COUNT,)


@pytest.mark.parametrize("temps", [np.array([220.0, 280.0, 330.0], dtype=np.float64)])
def test_bt_roundtrip_tests(temps):
    wavelengths = HYTES_WAVELENGTHS_NM[[0, HYTES_BAND_COUNT // 2, HYTES_BAND_COUNT - 1]]
    radiance = bt_to_radiance(temps, wavelengths)
    recovered = radiance_to_bt(radiance, wavelengths)
    np.testing.assert_allclose(recovered, temps, rtol=1e-6, atol=1e-6)
