import numpy as np
import pytest

pytest.importorskip("rasterio")
import rasterio

from alchemi.types import SpectrumKind
from alchemi_hsi.io.emit import emit_pixel, load_emit_l1b


@pytest.fixture()
def synthetic_emit_cube(tmp_path):
    width, height, bands = 3, 2, 5
    wavelengths_um = np.array([0.4, 0.5, 1.42, 1.92, 2.45], dtype=np.float32)
    radiance_um = np.arange(width * height * bands, dtype=np.float32)
    radiance_um = radiance_um.reshape(bands, height, width) + 1.0
    path = tmp_path / "emit_l1b.tif"

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=bands,
        dtype=radiance_um.dtype,
    ) as dst:
        dst.write(radiance_um)
        dst.update_tags(
            wavelengths=",".join(map(str, wavelengths_um)),
            wavelength_units="micrometers",
            radiance_units="W/m^2/sr/um",
        )

    return path, wavelengths_um, radiance_um


def test_wavelengths_monotonic_emit(synthetic_emit_cube):
    path, _, _ = synthetic_emit_cube
    ds = load_emit_l1b(str(path))

    wavelengths_nm = ds.coords["wavelength_nm"].values
    assert np.all(np.diff(wavelengths_nm) > 0)
    assert wavelengths_nm[0] >= 380
    assert wavelengths_nm[-1] <= 2500


def test_units_conversion_nm_emit(synthetic_emit_cube):
    path, _, radiance_um = synthetic_emit_cube
    ds = load_emit_l1b(str(path))

    converted = radiance_um / 1000.0
    loaded = ds["radiance"].values
    np.testing.assert_allclose(loaded, np.moveaxis(converted, 0, -1))
    assert ds["radiance"].attrs["units"] == "W·m⁻²·sr⁻¹·nm⁻¹"


def test_mask_water_bands_emit(synthetic_emit_cube):
    path, _, _ = synthetic_emit_cube
    ds = load_emit_l1b(str(path), band_mask=True)

    mask = ds["band_mask"].values
    assert mask.shape == (ds.dims["band"],)
    assert mask.tolist() == [True, True, False, False, True]

    ds_all = load_emit_l1b(str(path), band_mask=False)
    assert ds_all["band_mask"].values.all()


def test_emit_pixel_returns_spectrum(synthetic_emit_cube):
    path, _, _ = synthetic_emit_cube
    ds = load_emit_l1b(str(path))

    spectrum = emit_pixel(ds, 1, 0)
    assert spectrum.kind is SpectrumKind.RADIANCE
    assert spectrum.units == "W·m⁻²·sr⁻¹·nm⁻¹"
    np.testing.assert_allclose(spectrum.values, ds["radiance"].isel(y=1, x=0).values)
    assert np.array_equal(spectrum.wavelengths.nm, ds.coords["wavelength_nm"].values)
    assert np.array_equal(spectrum.mask, ds["band_mask"].values)
