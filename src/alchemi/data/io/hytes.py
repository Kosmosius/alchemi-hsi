"""Helpers for loading HyTES Level-1B brightness-temperature products."""

# mypy: ignore-errors

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import xarray as xr

from alchemi.types import Spectrum, SpectrumKind, WavelengthGrid

HYTES_BAND_COUNT = 256
HYTES_WAVELENGTH_MIN_NM = 7_500.0
HYTES_WAVELENGTH_MAX_NM = 12_000.0

HYTES_WAVELENGTHS_NM = np.linspace(
    HYTES_WAVELENGTH_MIN_NM,
    HYTES_WAVELENGTH_MAX_NM,
    HYTES_BAND_COUNT,
    dtype=np.float64,
)

_BT_VAR_CANDIDATES: tuple[str, ...] = (
    "brightness_temp",
    "brightness_temperature",
    "BrightnessTemperature",
    "bt",
    "BT_L1",
)
_BAND_DIM_CANDIDATES: tuple[str, ...] = (
    "band",
    "bands",
    "wavelength",
    "wavelengths",
    "spectral_band",
    "channel",
)
_Y_DIM_CANDIDATES: tuple[str, ...] = (
    "y",
    "row",
    "rows",
    "line",
    "along_track",
)
_X_DIM_CANDIDATES: tuple[str, ...] = (
    "x",
    "col",
    "cols",
    "column",
    "sample",
    "cross_track",
)

__all__ = [
    "HYTES_BAND_COUNT",
    "HYTES_WAVELENGTHS_NM",
    "hytes_pixel_bt",
    "load_hytes_l1b_bt",
]


def load_hytes_l1b_bt(path: str | Path) -> xr.Dataset:
    """Load a HyTES L1B brightness-temperature cube.

    Parameters
    ----------
    path:
        Path to the HyTES L1B product readable by :func:`xarray.open_dataset`.

    Returns
    -------
    xarray.Dataset
        Dataset with dimensions ``(y, x, band)`` that exposes a ``brightness_temp``
        variable in Kelvin and a ``wavelength_nm`` coordinate expressed in nanometres.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    src = xr.open_dataset(path)
    try:
        data = src.load()
    finally:
        close = getattr(src, "close", None)
        if callable(close):
            close()

    bt_var = _find_bt_variable(data)
    bt = data[bt_var]

    band_dim = _find_dim(bt.dims, _BAND_DIM_CANDIDATES)
    if band_dim is None:
        msg = "Unable to identify spectral dimension in HyTES dataset"
        raise ValueError(msg)

    spatial_dims = [dim for dim in bt.dims if dim != band_dim]
    if len(spatial_dims) != 2:
        msg = "HyTES brightness temperature must have exactly two spatial dimensions"
        raise ValueError(msg)

    y_dim = _find_dim(spatial_dims, _Y_DIM_CANDIDATES) or spatial_dims[0]
    x_dim = _find_dim(spatial_dims, _X_DIM_CANDIDATES)
    if x_dim is None or x_dim == y_dim:
        # Pick the remaining dimension if automatic detection failed.
        remaining = [dim for dim in spatial_dims if dim != y_dim]
        if not remaining:
            msg = "Unable to determine cross-track dimension for HyTES dataset"
            raise ValueError(msg)
        x_dim = remaining[0]

    bt = bt.transpose(y_dim, x_dim, band_dim)
    bt = bt.rename({y_dim: "y", x_dim: "x", band_dim: "band"})

    bt_values = np.asarray(bt.values, dtype=np.float64)
    units = _normalise_unit_string(bt.attrs.get("units") or data.attrs.get("bt_units"))
    bt_values = _ensure_kelvin(bt_values, units)

    sizes = {dim: size for dim, size in zip(bt.dims, bt.values.shape, strict=True)}
    y_coord = _extract_coordinate(bt, "y", sizes["y"])
    x_coord = _extract_coordinate(bt, "x", sizes["x"])
    band_coord = np.arange(sizes["band"], dtype=np.int32)

    coords = {
        "y": y_coord,
        "x": x_coord,
        "band": band_coord,
        "wavelength_nm": ("band", HYTES_WAVELENGTHS_NM.copy()),
    }

    brightness_temp = xr.DataArray(
        bt_values,
        dims=("y", "x", "band"),
        coords=coords,
        attrs={"units": "K"},
    )

    band_mask = xr.DataArray(
        np.ones(sizes["band"], dtype=bool),
        dims=("band",),
        coords={"band": coords["band"]},
    )

    ds = xr.Dataset(
        data_vars={
            "brightness_temp": brightness_temp,
            "band_mask": band_mask,
        },
        coords=coords,
    )
    ds.attrs["sensor"] = "HyTES"
    ds.attrs["brightness_temp_units"] = "K"

    ds.coords["wavelength_nm"].attrs["units"] = "nm"
    return ds


def hytes_pixel_bt(ds: xr.Dataset, y: int, x: int) -> Spectrum:
    """Extract a HyTES pixel as a :class:`~alchemi.types.Spectrum` in brightness temperature."""

    if "brightness_temp" not in ds.data_vars or "wavelength_nm" not in ds.coords:
        msg = "Dataset must contain 'brightness_temp' variable and 'wavelength_nm' coordinate"
        raise KeyError(msg)

    bt = ds["brightness_temp"].sel(y=y, x=x)
    values = np.asarray(bt.values, dtype=np.float64)

    units = bt.attrs.get("units") or ds.attrs.get("brightness_temp_units") or "K"
    units = _normalise_unit_string(units)
    values = _ensure_kelvin(values, units)

    wavelengths = np.asarray(ds.coords["wavelength_nm"].values, dtype=np.float64)

    mask = None
    if "band_mask" in ds.data_vars:
        mask = np.asarray(ds["band_mask"].values, dtype=bool)

    spectrum = Spectrum(
        wavelengths=WavelengthGrid(wavelengths),
        values=values,
        kind=SpectrumKind.BT,
        units="K",
        mask=mask,
        meta={
            "sensor": ds.attrs.get("sensor", "HyTES"),
            "y": int(ds.coords["y"].values[y]) if "y" in ds.coords else int(y),
            "x": int(ds.coords["x"].values[x]) if "x" in ds.coords else int(x),
        },
    )
    return spectrum


def _find_bt_variable(ds: xr.Dataset) -> str:
    for name in _BT_VAR_CANDIDATES:
        if name in ds.data_vars:
            return name
    if ds.data_vars:
        raise KeyError("Unable to locate brightness temperature variable in HyTES dataset")
    raise KeyError("HyTES dataset contains no variables")


def _find_dim(dims: Iterable[str], candidates: Iterable[str]) -> str | None:
    for candidate in candidates:
        if candidate in dims:
            return candidate
    return None


def _extract_coordinate(array: xr.DataArray, name: str, length: int) -> np.ndarray:
    coord = array.coords.get(name) if hasattr(array, "coords") else None
    if coord is not None:
        return np.asarray(coord.values, dtype=np.float64)
    return np.arange(length, dtype=np.float64)


def _normalise_unit_string(units: str | None) -> str | None:
    if units is None:
        return None
    return units.strip().lower()


def _ensure_kelvin(values: np.ndarray, units: str | None) -> np.ndarray:
    if units is None or units in {"k", "kelvin"}:
        return values.astype(np.float64, copy=False)
    if units in {"c", "celsius", "degc", "degree_celsius"}:
        return values.astype(np.float64, copy=False) + 273.15
    if units in {"f", "fahrenheit"}:
        return (values.astype(np.float64, copy=False) - 32.0) * (5.0 / 9.0) + 273.15
    msg = f"Unsupported brightness temperature units: {units!r}"
    raise ValueError(msg)
