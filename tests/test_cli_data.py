from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from alchemi.cli import app
from alchemi.data.cube import Cube


def _synthetic_cube() -> Cube:
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    axis_coords = {
        "y": np.array([0, 1], dtype=np.int32),
        "x": np.array([10, 11, 12], dtype=np.int32),
        "band": np.arange(4, dtype=np.int32),
    }
    wavelength_nm = np.linspace(400, 700, 4, dtype=np.float64)
    band_mask = np.array([True, False, True, True])
    metadata = {"mission": "synthetic"}
    return Cube(
        sensor="synthetic",
        quantity="radiance",
        values=values,
        axes=("y", "x", "band"),
        units="W·m⁻²·sr⁻¹·nm⁻¹",
        axis_coords=axis_coords,
        wavelength_nm=wavelength_nm,
        band_mask=band_mask,
        metadata=metadata,
    )


def test_data_info_prints_summary(monkeypatch):
    cube = _synthetic_cube()
    runner = CliRunner()

    monkeypatch.setattr("alchemi.cli._load_cube", lambda _: cube)

    result = runner.invoke(app, ["data", "info", "cube.dat"])
    assert result.exit_code == 0
    assert "Sensor: synthetic" in result.stdout
    assert "Shape: y=2, x=3, band=4" in result.stdout
    assert "Spectral range: 400.00–700.00 nm" in result.stdout
    assert "Band mask: 3/4 bands marked good" in result.stdout


def test_data_to_canonical_npz(monkeypatch, tmp_path):
    cube = _synthetic_cube()
    runner = CliRunner()

    monkeypatch.setattr("alchemi.cli._load_cube", lambda _: cube)

    src = tmp_path / "sample.h5"
    src.write_text("")

    result = runner.invoke(app, ["data", "to-canonical", str(src), "--out", "npz"])
    assert result.exit_code == 0

    dest = src.with_name("sample.canonical.npz")
    assert dest.exists()

    data = np.load(dest, allow_pickle=False)
    np.testing.assert_allclose(data["values"], cube.values)
    attrs = json.loads(str(data["attrs_json"]))
    assert attrs["sensor"] == "synthetic"
    assert attrs["units"] == cube.units


def test_data_to_canonical_zarr(monkeypatch, tmp_path):
    zarr = pytest.importorskip("zarr")
    cube = _synthetic_cube()
    runner = CliRunner()

    monkeypatch.setattr("alchemi.cli._load_cube", lambda _: cube)

    src = tmp_path / "sample2.h5"
    src.write_text("")

    result = runner.invoke(app, ["data", "to-canonical", str(src), "--out", "zarr"])
    assert result.exit_code == 0

    dest = src.with_name("sample2.canonical.zarr")
    root = zarr.open_group(str(dest), mode="r")
    np.testing.assert_allclose(root["values"], cube.values)
    assert root.attrs["sensor"] == "synthetic"
    assert root.attrs["units"] == cube.units
    np.testing.assert_allclose(root["wavelength_nm"], cube.wavelength_nm)
    np.testing.assert_array_equal(root["band_mask"], cube.band_mask)

    axes_group = root["axes"]
    for axis in cube.axes:
        np.testing.assert_allclose(axes_group[axis], cube.axis_coords[axis])
