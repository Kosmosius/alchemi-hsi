from __future__ import annotations

import json

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
    band_mask = np.array([True, False, True, True])
    wavelength_nm = np.linspace(400, 700, 4, dtype=np.float64)
    return Cube(
        data=values,
        axis=wavelength_nm,
        axis_unit="wavelength_nm",
        value_kind="radiance",
        axis_names=("y", "x", "band"),
        axis_coords=axis_coords,
        band_mask=band_mask,
        sensor="synthetic",
        units="W·m⁻²·sr⁻¹·nm⁻¹",
        attrs={"mission": "synthetic"},
    )


def test_data_info_prints_summary(monkeypatch):
    cube = _synthetic_cube()
    runner = CliRunner()

    monkeypatch.setattr("alchemi.cli._load_cube", lambda _path, _sensor=None: cube)

    result = runner.invoke(app, ["data", "info", "cube.dat"])
    assert result.exit_code == 0
    assert "Sensor: synthetic" in result.stdout
    assert "Shape: y=2, x=3, band=4" in result.stdout
    assert "Spectral range: 400.00-700.00 nm" in result.stdout
    assert "Band mask: 3/4 bands marked good" in result.stdout


def test_data_to_canonical_npz(monkeypatch, tmp_path):
    cube = _synthetic_cube()
    runner = CliRunner()

    monkeypatch.setattr("alchemi.cli._load_cube", lambda _path, _sensor=None: cube)

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

    monkeypatch.setattr("alchemi.cli._load_cube", lambda _path, _sensor=None: cube)

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


def test_data_info_sensor_override(monkeypatch, tmp_path):
    cube = _synthetic_cube()
    runner = CliRunner()

    def fail_sniff(_path):
        raise AssertionError("sniffer should be bypassed when sensor is provided")

    def fake_load_emit(path):
        fake_load_emit.called = path  # type: ignore[attr-defined]
        return "dataset"

    monkeypatch.setattr("alchemi.cli._sniff_dataset", fail_sniff)
    monkeypatch.setattr("alchemi.cli.load_emit_l1b", fake_load_emit)
    monkeypatch.setattr("alchemi.cli.Cube.from_xarray", lambda _ds: cube)

    src = tmp_path / "mystery.bin"
    src.write_text("")

    result = runner.invoke(app, ["data", "info", str(src), "--sensor", "emit"])
    assert result.exit_code == 0
    assert getattr(fake_load_emit, "called") == str(src)
    assert "Sensor: synthetic" in result.stdout


def test_data_info_sniff_failure_message(monkeypatch, tmp_path):
    runner = CliRunner()

    monkeypatch.setattr("alchemi.cli._sniff_dataset", lambda _path: None)

    src = tmp_path / "unknown.bin"
    src.write_text("")

    result = runner.invoke(app, ["data", "info", str(src)])
    assert result.exit_code != 0

    output = (result.stderr or "") + (result.stdout or "")
    assert "--sensor" in output
    assert "emit, enmap, avirisng, hytes, mako" in output
