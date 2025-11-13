"""Loader utilities for AVIRIS-NG Level-1B products."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

import xarray as xr
from alchemi.types import Spectrum, SpectrumKind, WavelengthGrid

__all__ = ["avirisng_pixel", "load_avirisng_l1b"]

_RAD_UNITS = "W·m^-2·sr^-1·nm^-1"


def load_avirisng_l1b(path: str | Path) -> xr.Dataset:
    """Load an AVIRIS-NG Level-1B cube."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    data = xr.open_dataset(path).load()
    radiance = _xr_find_variable(data, ["radiance", "Radiance", "L1B_Radiance"])
    if radiance is None:
        raise KeyError("Could not locate radiance variable in AVIRIS-NG file")
    radiance = _standardize_dims(radiance)

    wavelengths = _xr_find_variable(data, ["wavelength", "wavelength_nm", "wavelengths"])
    if wavelengths is None:
        raise KeyError("Could not locate wavelength information in AVIRIS-NG file")
    wavelengths_nm = _to_nanometers(wavelengths)

    fwhm = _xr_find_variable(data, ["fwhm", "fwhm_nm", "bandwidth"])
    fwhm_nm = None if fwhm is None else _to_nanometers(fwhm)

    mask = _xr_find_variable(data, ["band_mask", "bad_bands", "quality_flags"])
    mask_values = None if mask is None else np.asarray(mask.values, dtype=bool)

    radiance = _convert_radiance_units(radiance)

    radiance = radiance.transpose("y", "x", "band")
    radiance = radiance.astype(np.float64)
    radiance.attrs["units"] = _RAD_UNITS
    radiance.attrs["quantity"] = "radiance"

    dataset = xr.Dataset()
    dataset["radiance"] = radiance
    dataset = dataset.assign_coords(wavelength_nm=("band", wavelengths_nm))
    dataset.attrs.update(quantity="radiance", sensor="avirisng", units=_RAD_UNITS)

    if fwhm_nm is not None:
        dataset["fwhm_nm"] = ("band", np.asarray(fwhm_nm, dtype=np.float64))
    if mask_values is not None:
        dataset["band_mask"] = ("band", mask_values.astype(bool))

    return dataset


def avirisng_pixel(ds: xr.Dataset, y: int, x: int) -> Spectrum:
    """Extract a single pixel as a :class:`~alchemi.types.Spectrum`."""

    radiance = ds["radiance"].sel(y=y, x=x)
    wavelengths = ds["wavelength_nm"].values
    mask = ds.get("band_mask")
    fwhm = ds.get("fwhm_nm")

    meta = {"sensor": ds.attrs.get("sensor", "avirisng")}
    if fwhm is not None:
        meta["fwhm_nm"] = np.asarray(fwhm.values, dtype=np.float64)

    return Spectrum(
        wavelengths=WavelengthGrid(np.asarray(wavelengths, dtype=np.float64)),
        values=np.asarray(radiance.values, dtype=np.float64),
        kind=SpectrumKind.RADIANCE,
        units=ds.attrs.get("units", _RAD_UNITS),
        mask=np.asarray(mask.values, dtype=bool) if mask is not None else None,
        meta=meta,
    )


def _xr_find_variable(ds: xr.Dataset, names: Iterable[str]) -> xr.DataArray | None:
    for name in names:
        if name in ds.data_vars:
            return ds[name]
        if name in ds.coords:
            coord = ds.coords[name]
            return xr.DataArray(coord.values, dims=coord.dims, attrs=dict(coord.attrs))
    for name, var in ds.data_vars.items():
        if any(token in name.lower() for token in names):
            return var
    return None


def _standardize_dims(arr: xr.DataArray) -> xr.DataArray:
    dims = list(arr.dims)
    rename = {}
    if "y" not in dims:
        rename[dims[0]] = "y"
    if "x" not in dims:
        rename[dims[1]] = "x"
    if "band" not in dims:
        rename[dims[-1]] = "band"
    if rename:
        arr = arr.rename(rename)
    return arr


def _to_nanometers(arr: xr.DataArray) -> np.ndarray:
    data = np.asarray(arr.values, dtype=np.float64)
    units = arr.attrs.get("units", "")
    if units:
        units = units.lower()
    if units in {"um", "micrometer", "micrometers", "micron", "microns"}:
        data = data * 1000.0
    elif units in {"m"}:
        data = data * 1e9
    elif units in {"nm", "nanometer", "nanometers"}:
        data = data
    else:
        data = data.astype(np.float64)
    if data.ndim != 1:
        data = data.reshape(-1)
    if np.any(np.diff(np.sort(data)) <= 0):
        data = np.sort(data)
    return data.astype(np.float64)


def _convert_radiance_units(arr: xr.DataArray) -> xr.DataArray:
    units = arr.attrs.get("units", "")
    if units:
        units = units.replace("μ", "u").lower()
    if "w" in units and "nm" in units:
        return arr
    if "w" in units and "um" in units:
        data = np.asarray(arr.values, dtype=np.float64) / 1000.0
        out = xr.DataArray(data, dims=arr.dims, coords=arr.coords)
        out.attrs.update(arr.attrs)
        out.attrs["units"] = _RAD_UNITS
        return out
    if "mw" in units and "um" in units:
        data = np.asarray(arr.values, dtype=np.float64) * 1e-6
        out = xr.DataArray(data, dims=arr.dims, coords=arr.coords)
        out.attrs.update(arr.attrs)
        out.attrs["units"] = _RAD_UNITS
        return out
    out = arr.astype(np.float64)
    out.attrs.update(arr.attrs)
    out.attrs["units"] = _RAD_UNITS
    return out
