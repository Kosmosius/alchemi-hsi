"""Loader utilities for EnMAP Level-1B products."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr

from alchemi.types import Spectrum, SpectrumKind, WavelengthGrid

__all__ = ["load_enmap_l1b", "enmap_pixel"]

_RAD_UNITS = "W·m^-2·sr^-1·nm^-1"


@dataclass
class _SpectralSlice:
    radiance: xr.DataArray
    wavelengths_nm: np.ndarray
    fwhm_nm: Optional[np.ndarray]
    mask: Optional[np.ndarray]


def load_enmap_l1b(path_vnir: str | Path, path_swir: str | Path) -> xr.Dataset:
    """Load and merge the VNIR and SWIR cubes from EnMAP L1B products."""

    slice_vnir = _load_slice(Path(path_vnir))
    slice_swir = _load_slice(Path(path_swir))

    radiance = xr.concat([slice_vnir.radiance, slice_swir.radiance], dim="band")

    wavelengths = np.concatenate([slice_vnir.wavelengths_nm, slice_swir.wavelengths_nm])
    fwhm_all = None
    if slice_vnir.fwhm_nm is not None or slice_swir.fwhm_nm is not None:
        parts = [p for p in (slice_vnir.fwhm_nm, slice_swir.fwhm_nm) if p is not None]
        fwhm_all = np.concatenate(parts)
    mask_all = None
    if slice_vnir.mask is not None or slice_swir.mask is not None:
        parts = [p for p in (slice_vnir.mask, slice_swir.mask) if p is not None]
        mask_all = np.concatenate(parts)

    order = np.argsort(wavelengths)
    wavelengths = wavelengths[order]
    radiance_values = np.asarray(radiance.values)[:, :, order]
    coords = {}
    for dim in ("y", "x"):
        if dim in radiance.coords:
            coords[dim] = radiance.coords[dim]
        else:
            idx = 0 if dim == "y" else 1
            coords[dim] = np.arange(radiance_values.shape[idx], dtype=np.int64)
    coords["band"] = np.arange(radiance_values.shape[2], dtype=np.int64)
    radiance = xr.DataArray(
        radiance_values,
        dims=("y", "x", "band"),
        coords=coords,
    )
    radiance.attrs["units"] = _RAD_UNITS
    radiance.attrs["quantity"] = "radiance"

    if fwhm_all is not None:
        fwhm_all = fwhm_all[order]
    if mask_all is not None:
        mask_all = mask_all.astype(bool)[order]

    if np.any(np.diff(wavelengths) <= 0):
        raise ValueError("Merged wavelengths must be strictly increasing")

    dataset = xr.Dataset()
    dataset["radiance"] = radiance
    dataset = dataset.assign_coords(wavelength_nm=("band", wavelengths.astype(np.float64)))

    if fwhm_all is not None:
        dataset["fwhm_nm"] = ("band", fwhm_all.astype(np.float64))
    if mask_all is not None:
        dataset["band_mask"] = ("band", mask_all.astype(bool))

    dataset.attrs.update(quantity="radiance", sensor="enmap", units=_RAD_UNITS)

    return dataset


def enmap_pixel(ds: xr.Dataset, y: int, x: int) -> Spectrum:
    """Extract a single pixel as a :class:`~alchemi.types.Spectrum`."""

    radiance = ds["radiance"].sel(y=y, x=x)
    wavelengths = ds["wavelength_nm"].values
    mask = ds.get("band_mask")
    fwhm = ds.get("fwhm_nm")

    meta = {"sensor": ds.attrs.get("sensor", "enmap")}
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


def _load_slice(path: Path) -> _SpectralSlice:
    data = xr.open_dataset(path).load()

    rad = _find_radiance(data)
    rad, spectral_dim = _standardize_dims(rad)
    wavelengths = _extract_wavelengths(data, spectral_dim, rad)
    fwhm = _extract_fwhm(data, spectral_dim)
    mask = _extract_mask(data, spectral_dim)

    rad = rad.astype(np.float64)
    rad = _convert_radiance(rad)

    return _SpectralSlice(rad, wavelengths, fwhm, mask)


def _find_radiance(ds: xr.Dataset) -> xr.DataArray:
    if "radiance" in ds.data_vars:
        return ds["radiance"]

    candidates = []
    for var in ds.data_vars.values():
        if var.ndim != 3:
            continue
        dims = {d.lower() for d in var.dims}
        if {"y", "x"}.issubset(dims) or any("wave" in d or "band" in d for d in dims):
            candidates.append(var)
    if not candidates:
        raise ValueError("Could not find radiance cube in dataset")
    return candidates[0]


def _standardize_dims(arr: xr.DataArray) -> tuple[xr.DataArray, str]:
    rename = {}
    spectral_dim = None
    for dim in arr.dims:
        low = dim.lower()
        if low in {"rows", "row", "line", "lines", "y"}:
            rename[dim] = "y"
        elif low in {"cols", "col", "sample", "samples", "x"}:
            rename[dim] = "x"
        elif "band" in low or "wave" in low or low in {"spectral", "lambda"}:
            rename[dim] = "band"
            spectral_dim = dim
    arr = arr.rename(rename)
    missing = {"y", "x", "band"} - set(arr.dims)
    if missing:
        raise ValueError(f"Radiance array missing expected dimensions: {missing}")
    if spectral_dim is None:
        spectral_dim = "band"
    return arr.transpose("y", "x", "band"), spectral_dim


def _extract_wavelengths(ds: xr.Dataset, band_dim: str, arr: xr.DataArray) -> np.ndarray:
    band_coord = arr.coords.get("band")
    if band_coord is not None and band_coord.ndim == 1:
        return _to_nm(np.asarray(band_coord.values, dtype=np.float64), band_coord.attrs.get("units"))

    for name in list(ds.coords) + list(ds.data_vars):
        if name == band_dim:
            continue
        if "wave" in name.lower() or "lambda" in name.lower():
            var = ds[name]
            if band_dim in var.dims and var.ndim == 1:
                return _to_nm(np.asarray(var.values, dtype=np.float64), var.attrs.get("units"))

    fallback = ds.coords.get(band_dim)
    if fallback is not None and fallback.ndim == 1:
        return _to_nm(np.asarray(fallback.values, dtype=np.float64), fallback.attrs.get("units"))

    raise ValueError("Could not locate wavelength coordinate")


def _extract_fwhm(ds: xr.Dataset, band_dim: str) -> Optional[np.ndarray]:
    for name in list(ds.data_vars):
        low = name.lower()
        if "fwhm" in low or "bandwidth" in low:
            var = ds[name]
            if band_dim in var.dims and var.ndim == 1:
                return _to_nm(np.asarray(var.values, dtype=np.float64), var.attrs.get("units"))
    return None


def _extract_mask(ds: xr.Dataset, band_dim: str) -> Optional[np.ndarray]:
    for name in list(ds.data_vars):
        low = name.lower()
        if any(key in low for key in ["mask", "quality", "valid"]):
            var = ds[name]
            if band_dim in var.dims and var.ndim == 1:
                data = np.asarray(var.values)
                return data.astype(bool)
    return None


def _convert_radiance(arr: xr.DataArray) -> xr.DataArray:
    units = arr.attrs.get("units", "")
    units_l = units.lower()
    scale = _parse_scale(units_l)
    if _is_per_micrometer(units_l):
        scale *= 1e-3
    arr = xr.DataArray(arr.values * scale, coords=arr.coords, dims=arr.dims)
    arr.attrs["units"] = _RAD_UNITS
    return arr


def _parse_scale(units: str) -> float:
    if "mw" in units:
        return 1e-3
    if "uw" in units or "µw" in units or "micro" in units:
        return 1e-6
    return 1.0


def _is_per_micrometer(units: str) -> bool:
    tokens = ["um", "µm", "micrometer", "micrometre"]
    if not any(t in units for t in tokens):
        return False
    indicators = [
        "um-1",
        "µm-1",
        "per um",
        "per µm",
        "um^-1",
        "µm^-1",
        "micrometer-1",
        "micrometer^-1",
        "micrometre-1",
        "micrometre^-1",
    ]
    if any(ind in units for ind in indicators):
        return True
    return any(f"/{tok}" in units for tok in tokens)


def _to_nm(values: np.ndarray, unit: Optional[str]) -> np.ndarray:
    if unit is None:
        return _infer_nm(values)
    unit_l = unit.lower()
    if "um" in unit_l or "µm" in unit_l or "micrometer" in unit_l or "micrometre" in unit_l:
        return values * 1e3
    if "nm" in unit_l:
        return values
    return _infer_nm(values)


def _infer_nm(values: np.ndarray) -> np.ndarray:
    if np.nanmax(values) <= 10.0:
        return values * 1e3
    return values
