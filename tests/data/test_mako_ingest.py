"""Tests for the COMEX Mako Level-2S radiance ingestion utilities."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from alchemi.types import SpectrumKind
from alchemi.data.adapters.mako import load_mako_pixel
from alchemi.io import MAKO_BAND_COUNT, mako_pixel_radiance, open_mako_l2s

pytest.importorskip("rasterio")
import rasterio


@pytest.fixture()
def synthetic_mako_cube(tmp_path: Path) -> tuple[Path, np.ndarray, np.ndarray]:
    width, height = 2, 1
    data = np.linspace(1.0, MAKO_BAND_COUNT * width, MAKO_BAND_COUNT * width, dtype=np.float32)
    data = data.reshape(MAKO_BAND_COUNT, height, width)

    wavelengths_um = np.linspace(7.5, 13.5, MAKO_BAND_COUNT, dtype=np.float32)
    bbl = np.ones(MAKO_BAND_COUNT, dtype=int)
    bbl[-1] = 0

    cube_path = tmp_path / "S_TEST_TILE_L2S.dat"
    with rasterio.open(
        cube_path,
        "w",
        driver="ENVI",
        width=width,
        height=height,
        count=MAKO_BAND_COUNT,
        dtype="float32",
    ) as dst:
        dst.write(data)
        dst.update_tags(
            ns="ENVI",
            **{
                "wavelength": _format_envi_list(wavelengths_um),
                "wavelength units": "Micrometers",
                "bbl": _format_envi_list(bbl),
            },
        )

    return cube_path, data.astype(np.float64), wavelengths_um.astype(np.float64)


def test_mako_l2s_wavelengths_monotonic(synthetic_mako_cube: tuple[Path, np.ndarray, np.ndarray]):
    path, _, wavelengths_um = synthetic_mako_cube
    ds = open_mako_l2s(path)

    wavelengths_nm = ds.coords["wavelength_nm"].values
    assert np.all(np.diff(wavelengths_nm) > 0)
    assert math.isclose(wavelengths_nm[0], wavelengths_um[0] * 1000.0)
    assert math.isclose(wavelengths_nm[-1], wavelengths_um[-1] * 1000.0)


def test_mako_l2s_units_conversion_nm(synthetic_mako_cube: tuple[Path, np.ndarray, np.ndarray]):
    path, microflick_data, _ = synthetic_mako_cube
    ds = open_mako_l2s(path)

    radiance = ds["radiance"].isel(y=0, x=0)
    assert np.allclose(radiance.values, microflick_data[:, 0, 0] * 1e-5)

    spectrum = mako_pixel_radiance(ds, 0, 1)
    assert np.allclose(spectrum.values, microflick_data[:, 0, 1] * 1e-5)
    assert spectrum.units == "W·m⁻²·sr⁻¹·nm⁻¹"


def test_mako_l2s_bandcount_128(synthetic_mako_cube: tuple[Path, np.ndarray, np.ndarray]):
    path, _, _ = synthetic_mako_cube
    ds = open_mako_l2s(path)
    assert ds.sizes["band"] == MAKO_BAND_COUNT


def test_mako_l2s_sample_smoketest(synthetic_mako_cube: tuple[Path, np.ndarray, np.ndarray]):
    path, _, _ = synthetic_mako_cube
    ds = open_mako_l2s(path)

    sample = load_mako_pixel(ds, (0, 1))
    assert sample.meta["sensor"] == "mako"
    assert sample.meta["col"] == 1
    assert sample.spectrum.kind == SpectrumKind.RADIANCE


def _format_envi_list(values: np.ndarray) -> str:
    values_list = ",\n ".join(str(v) for v in values)
    return "{" + values_list + "}"

