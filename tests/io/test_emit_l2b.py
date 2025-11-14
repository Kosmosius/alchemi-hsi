from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from alchemi.io.emit_l2b import (
    EMIT_TO_USGS,
    iter_high_confident_pixels,
    load_emit_l2b,
    map_emit_group_to_splib,
)


@pytest.fixture()
def synthetic_emit_l2b(tmp_path: Path) -> Path:
    y_size, x_size = 3, 4
    minerals = np.array(
        [
            ["ILLITE_MUSCOVITE_GROUP", "", np.nan, "KAOLINITE_GROUP"],
            ["SMECTITE_GROUP", "ILLITE_MUSCOVITE_GROUP", "KAOLINITE_GROUP", None],
            ["ALUNITE_GROUP", "UNKNOWN_GROUP", "", "ILLITE_MUSCOVITE_GROUP"],
        ],
        dtype=object,
    )
    fit_r2 = np.array(
        [
            [0.95, 0.50, np.nan, 0.99],
            [0.92, 0.88, 0.91, 0.93],
            [0.70, 0.96, 0.85, 1.00],
        ],
        dtype=float,
    )
    band_depth_2200 = np.arange(y_size * x_size, dtype=float).reshape(y_size, x_size)
    band_depth_2330 = band_depth_2200 + 0.5

    ds = xr.Dataset(
        data_vars={
            "mineral_group": (("y", "x"), minerals),
            "fit_r2": (("y", "x"), fit_r2),
            "band_depth_2200": (("y", "x"), band_depth_2200),
            "band_depth_2330": (("y", "x"), band_depth_2330),
            "all_nan": (("y", "x"), np.full((y_size, x_size), np.nan)),
        },
        coords={"y": np.arange(y_size), "x": np.arange(x_size)},
        attrs={"description": "Synthetic EMIT L2B MIN cube for testing"},
    )

    path = tmp_path / "emit_l2b.nc"
    ds.to_netcdf(path)
    return path


def test_load_emit_l2b_dimensions_and_variables(synthetic_emit_l2b: Path) -> None:
    ds = load_emit_l2b(synthetic_emit_l2b)

    assert ds.sizes["y"] == 3
    assert ds.sizes["x"] == 4
    assert "mineral_group" in ds
    assert "fit_r2" in ds
    assert "band_depth_2200" in ds
    # ``drop_empty=True`` removes variables that are entirely NaN.
    assert "all_nan" not in ds


def test_load_emit_l2b_keep_empty_variable(synthetic_emit_l2b: Path) -> None:
    ds = load_emit_l2b(synthetic_emit_l2b, drop_empty=False)
    assert "all_nan" in ds


def test_load_emit_l2b_missing_mineral_variable(tmp_path: Path) -> None:
    ds = xr.Dataset(
        data_vars={
            "fit_r2": (("y", "x"), np.ones((2, 2))),
            "band_depth_2200": (("y", "x"), np.ones((2, 2))),
        },
        coords={"y": np.arange(2), "x": np.arange(2)},
    )
    path = tmp_path / "missing.nc"
    ds.to_netcdf(path)

    with pytest.raises(ValueError):
        load_emit_l2b(path)


def test_iter_high_confident_pixels_filters_and_labels(
    synthetic_emit_l2b: Path,
) -> None:
    ds = load_emit_l2b(synthetic_emit_l2b)

    labels = list(iter_high_confident_pixels(ds, r2_min=0.9))

    expected = [
        (0, 0, "ILLITE_MUSCOVITE_GROUP", pytest.approx(0.95)),
        (0, 3, "KAOLINITE_GROUP", pytest.approx(0.99)),
        (1, 0, "SMECTITE_GROUP", pytest.approx(0.92)),
        (1, 2, "KAOLINITE_GROUP", pytest.approx(0.91)),
        (2, 1, "UNKNOWN_GROUP", pytest.approx(0.96)),
        (2, 3, "ILLITE_MUSCOVITE_GROUP", pytest.approx(1.00)),
    ]
    # These labels are *weak supervision*: we only check that the iterator picks
    # the high-confidence mineral hypotheses and skips missing entries.
    assert labels == expected


@pytest.mark.parametrize(
    "group_name, expected",
    [
        ("ILLITE_MUSCOVITE_GROUP", EMIT_TO_USGS["ILLITE_MUSCOVITE_GROUP"]),
        ("KAOLINITE_GROUP", EMIT_TO_USGS["KAOLINITE_GROUP"]),
        ("UNKNOWN_GROUP", []),
    ],
)
def test_map_emit_group_to_splib(group_name: str, expected: list[str]) -> None:
    assert map_emit_group_to_splib(group_name) == expected
