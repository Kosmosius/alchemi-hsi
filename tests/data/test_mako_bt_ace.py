"""Tests for the COMEX Mako Level-3 BTEMP and ACE ingestion utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from alchemi.data.adapters.mako import load_mako_pixel_bt
from alchemi.io import ACE_GAS_NAMES, MAKO_BAND_COUNT, mako_pixel_bt, open_mako_ace, open_mako_btemp
from alchemi.types import SpectrumKind

rasterio = pytest.importorskip("rasterio")


@pytest.fixture()
def synthetic_btemp_cube(tmp_path: Path) -> tuple[Path, np.ndarray]:
    width, height = 2, 1
    celsius_data = np.linspace(
        10.0, MAKO_BAND_COUNT * width + 10.0, MAKO_BAND_COUNT * width, dtype=np.float32
    )
    celsius_data = celsius_data.reshape(MAKO_BAND_COUNT, height, width)

    wavelengths_um = np.linspace(7.5, 13.5, MAKO_BAND_COUNT, dtype=np.float32)

    cube_path = tmp_path / "S_TEST_TILE_BTEMP.dat"
    with rasterio.open(
        cube_path,
        "w",
        driver="ENVI",
        width=width,
        height=height,
        count=MAKO_BAND_COUNT,
        dtype="float32",
    ) as dst:
        dst.write(celsius_data)
        dst.update_tags(
            ns="ENVI",
            **{
                "wavelength": _format_envi_list(wavelengths_um),
                "wavelength units": "Micrometers",
            },
        )

    return cube_path, celsius_data.astype(np.float64)


@pytest.fixture()
def synthetic_ace_cube(tmp_path: Path) -> tuple[Path, np.ndarray]:
    width, height = 3, 2
    band_count = len(ACE_GAS_NAMES)
    ace_data = np.linspace(-1.0, 1.0, band_count * width * height, dtype=np.float32)
    ace_data = ace_data.reshape(band_count, height, width)

    cube_path = tmp_path / "S_TEST_TILE_ACE.dat"
    with rasterio.open(
        cube_path,
        "w",
        driver="ENVI",
        width=width,
        height=height,
        count=band_count,
        dtype="float32",
    ) as dst:
        dst.write(ace_data)

    return cube_path, ace_data.astype(np.float64)


def test_mako_btemp_units_and_shape(synthetic_btemp_cube: tuple[Path, np.ndarray]):
    path, celsius_data = synthetic_btemp_cube
    ds = open_mako_btemp(path)

    bt = ds["bt"].isel(y=0, x=0)
    expected = celsius_data[:, 0, 0] + 273.15
    assert np.allclose(bt.values, expected)
    assert ds.sizes["band"] == MAKO_BAND_COUNT

    spectrum = mako_pixel_bt(ds, 0, 1)
    assert spectrum.kind == SpectrumKind.BT
    assert np.allclose(spectrum.values, celsius_data[:, 0, 1] + 273.15)
    assert spectrum.units == "K"

    sample = load_mako_pixel_bt(ds, (0, 1))
    assert sample.spectrum.kind == SpectrumKind.BT
    assert sample.meta["sensor"] == "mako"


def test_mako_ace_band_order_smoketest(synthetic_ace_cube: tuple[Path, np.ndarray]):
    path, ace_data = synthetic_ace_cube
    ds = open_mako_ace(path)

    assert ds.sizes["gas_band"] == len(ACE_GAS_NAMES)
    assert list(ds["gas_name"].values) == ACE_GAS_NAMES

    matches = np.where(ds["gas_name"].values == "CH4")[0]
    methane_index = int(matches[0]) if matches.size else 0
    methane_map = ds["ace"].isel(gas_band=methane_index).values
    expected = np.moveaxis(ace_data, 0, -1)[:, :, ACE_GAS_NAMES.index("CH4")]
    assert np.allclose(methane_map, expected)


def test_mako_btemp_quicklook_notebook_smoketest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    notebook_path = Path("notebooks/mako_comex_quicklook.ipynb")
    monkeypatch.setenv("MAKO_COMEX_QUICKLOOK_MODE", "synthetic")

    with notebook_path.open("r", encoding="utf-8") as stream:
        notebook = json.load(stream)

    namespace: dict[str, object] = {"__name__": "__main__"}
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue
        compiled = compile(source, str(notebook_path), "exec")
        exec(compiled, namespace)


def _format_envi_list(values: np.ndarray) -> str:
    values_list = ",\n ".join(str(v) for v in values)
    return "{" + values_list + "}"
