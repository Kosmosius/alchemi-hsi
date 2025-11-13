"""Helpers for loading HyTES Level-1B brightness-temperature products."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import xarray as xr
from alchemi.types import Spectrum, SpectrumKind, WavelengthGrid

HYTES_BAND_COUNT = 256
HYTES_WAVELENGTHS_NM = np.linspace(7600.0, 11900.0, HYTES_BAND_COUNT)

__all__ = [
    "HYTES_BAND_COUNT",
    "HYTES_WAVELENGTHS_NM",
    "hytes_pixel_bt",
    "load_hytes_l1b_bt",
]


def load_hytes_l1b_bt(path: str | Path) -> xr.Dataset:
    """Load a HyTES brightness-temperature Level-1B product."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    data = xr.open_dataset(path).load()
    bt = _extract_bt(data)
    bt = _standardize_dims(bt)
    bt = bt.astype(np.float64)
    bt.attrs["units"] = "K"
    bt.attrs["quantity"] = "brightness_temp"

    dataset = xr.Dataset()
    dataset["brightness_temp"] = bt

    wavelengths = _extract_wavelengths(data, bt)
    dataset = dataset.assign_coords(wavelength_nm=("band", wavelengths))
    dataset.attrs.update(sensor="hytes", quantity="brightness_temp", units="K")

    band_axis = bt.dims.index("band") if "band" in bt.dims else -1
    band_size = bt.values.shape[band_axis]
    mask = _extract_mask(data, "band", band_size)
    if mask is not None:
        dataset["band_mask"] = ("band", mask)

    return dataset


def hytes_pixel_bt(ds: xr.Dataset, y: int, x: int) -> Spectrum:
    """Return a brightness-temperature :class:`~alchemi.types.Spectrum` for a pixel."""

    bt = ds["brightness_temp"].sel(y=y, x=x)
    mask = ds.get("band_mask")
    spectrum = Spectrum(
        wavelengths=WavelengthGrid(np.asarray(ds["wavelength_nm"].values, dtype=np.float64)),
        values=np.asarray(bt.values, dtype=np.float64),
        kind=SpectrumKind.BT,
        units=ds.attrs.get("units", "K"),
        mask=np.asarray(mask.values, dtype=bool) if mask is not None else None,
        meta={"sensor": ds.attrs.get("sensor", "hytes")},
    )
    return spectrum


def _extract_bt(ds: xr.Dataset) -> xr.DataArray:
    candidates = [
        name
        for name in ds.data_vars
        if "temp" in name.lower() or "brightness" in name.lower()
    ]
    if not candidates:
        raise KeyError("Could not locate brightness temperature variable")
    return ds[candidates[0]]


def _standardize_dims(arr: xr.DataArray) -> xr.DataArray:
    dims = list(arr.dims)
    rename = {}
    if "y" not in dims:
        rename[dims[0]] = "y"
    if "x" not in dims:
        rename[dims[1]] = "x"
    if "band" not in dims:
        rename[dims[-1]] = "band"
    return arr.rename(rename) if rename else arr


def _extract_wavelengths(ds: xr.Dataset, bt: xr.DataArray) -> np.ndarray:
    coord = None
    for name in bt.coords:
        if "wave" in name.lower():
            coord = bt.coords[name]
            break
    if coord is None:
        coord = _find_coord(ds, {"wavelength", "wavelength_nm", "wavenumber", "wavenumber_cm"})
    if coord is None:
        return HYTES_WAVELENGTHS_NM.copy()

    data = np.asarray(coord.values, dtype=np.float64)
    units = coord.attrs.get("units", "")
    units = units.replace("μ", "u").lower()
    if units in {"um", "micrometer", "micrometers", "micron", "microns"}:
        data = data * 1000.0
    elif units in {"cm^-1", "cm-1", "wavenumber"}:
        # Convert from wavenumber (1/cm) to wavelength in nm
        data = (1.0 / data) * 1e7
    return data.astype(np.float64)


def _extract_mask(ds: xr.Dataset, band_dim: str, band_size: int) -> np.ndarray | None:
    for name, var in ds.data_vars.items():
        if "mask" not in name.lower() and "quality" not in name.lower():
            continue

        if band_dim not in var.dims:
            continue

        values = np.asarray(var.values)
        axis = var.dims.index(band_dim)
        if values.shape[axis] != band_size:
            continue

        mask = np.asarray(values, dtype=bool)
        if mask.ndim == 1:
            return mask

        reduce_axes = tuple(idx for idx in range(mask.ndim) if idx != axis)
        if not reduce_axes:
            return mask.reshape(band_size)

        reduced = np.all(mask, axis=reduce_axes)
        return np.asarray(reduced, dtype=bool)
    return None


def _find_coord(ds: xr.Dataset, names: set[str]) -> xr.DataArray | None:
    for name in ds.coords:
        if name.lower() in names:
            return ds.coords[name]
    return None
