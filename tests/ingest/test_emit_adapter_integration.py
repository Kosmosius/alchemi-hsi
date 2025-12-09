"""End-to-end integration test for the EMIT ingest adapter."""

from __future__ import annotations

import numpy as np
import pytest

from alchemi.data.io import load_emit_l1b
from alchemi.ingest.emit import from_emit_l1b
from alchemi.physics.rad_reflectance import radiance_to_toa_reflectance
from alchemi.physics.solar import esun_for_sample
from alchemi.registry import get_srf

rasterio = pytest.importorskip("rasterio")

pytestmark = pytest.mark.physics_and_metadata


def _write_emit_fixture(path) -> tuple[np.ndarray, np.ndarray]:
    width, height, bands = 3, 2, 6
    wavelengths_um = np.array([0.42, 0.65, 1.42, 1.93, 2.25, 2.5], dtype=np.float32)
    radiance_um = np.linspace(1.0, 10.0, num=width * height * bands, dtype=np.float32)
    radiance_um = radiance_um.reshape(bands, height, width)

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

    return wavelengths_um, radiance_um


def test_emit_ingest_respects_physics_and_metadata(tmp_path):
    path = tmp_path / "emit_l1b.tif"
    wavelengths_um, _ = _write_emit_fixture(path)

    dataset = load_emit_l1b(path)
    cube = from_emit_l1b(dataset)

    height, width, band_count = cube.shape
    assert height > 0 and width > 0
    assert band_count == cube.axis.shape[0] == dataset["wavelength_nm"].shape[0]
    assert np.all(np.diff(cube.axis) > 0)
    assert cube.value_kind.value == "radiance"

    assert cube.srf_id is not None
    srf = get_srf(str(cube.srf_id))
    assert srf.centers_nm.size >= band_count

    assert cube.band_mask is not None
    assert cube.band_mask.shape == (band_count,)
    np.testing.assert_array_equal(cube.band_mask, dataset["band_mask"].values)

    sample = cube.sample_at(0, 0)
    np.testing.assert_array_equal(sample.spectrum.mask, cube.band_mask)

    esun_band = esun_for_sample(sample, mode="interp")
    reflectance = radiance_to_toa_reflectance(
        sample.spectrum, esun_band=esun_band, d_au=1.0, solar_zenith_deg=30.0
    )
    valid_reflectance = reflectance.values[cube.band_mask]
    p1, p99 = np.nanpercentile(valid_reflectance, [1, 99])
    assert p1 >= 0.0
    assert p99 <= 1.2
