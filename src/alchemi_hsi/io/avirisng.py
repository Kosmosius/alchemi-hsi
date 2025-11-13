"""Loader utilities for AVIRIS-NG Level-1B products."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import xarray as xr

from alchemi.types import Spectrum, SpectrumKind, WavelengthGrid

__all__ = ["load_avirisng_l1b", "avirisng_pixel"]

_RAD_UNITS = "W·m^-2·sr^-1·nm^-1"
_WATER_VAPOR_WINDOWS_NM: Sequence[tuple[float, float]] = (
    (1340.0, 1460.0),
    (1790.0, 1960.0),
)


@dataclass
class _SpectralData:
    radiance: np.ndarray  # [y, x, band]
    wavelengths_nm: np.ndarray  # [band]
    fwhm_nm: Optional[np.ndarray]
    band_mask: Optional[np.ndarray]


def load_avirisng_l1b(path: str | Path) -> xr.Dataset:
    """Load an AVIRIS-NG L1B radiance cube into an :class:`xarray.Dataset`."""

    spectral = _read_file(Path(path))

    order = np.argsort(spectral.wavelengths_nm)
    wavelengths = spectral.wavelengths_nm[order]
    radiance = spectral.radiance[..., order]

    y_size, x_size, band_size = radiance.shape
    coords = {
        "y": np.arange(y_size, dtype=np.int64),
        "x": np.arange(x_size, dtype=np.int64),
        "band": np.arange(band_size, dtype=np.int64),
    }

    dataset = xr.Dataset()
    dataset["radiance"] = xr.DataArray(
        radiance,
        dims=("y", "x", "band"),
        coords=coords,
        attrs={"units": _RAD_UNITS, "quantity": "radiance"},
    )
    dataset = dataset.assign_coords(wavelength_nm=("band", wavelengths.astype(np.float64)))

    if spectral.fwhm_nm is not None:
        dataset["fwhm_nm"] = ("band", np.asarray(spectral.fwhm_nm, dtype=np.float64)[order])

    mask = spectral.band_mask
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape[0] != band_size:
            raise ValueError("Band mask length must match number of bands")
        dataset["band_mask"] = ("band", mask[order])

    dataset.attrs.update(quantity="radiance", sensor="avirisng", units=_RAD_UNITS)

    if np.any(np.diff(wavelengths) <= 0):
        raise ValueError("Wavelengths must be strictly increasing")

    return dataset


def avirisng_pixel(ds: xr.Dataset, y: int, x: int) -> Spectrum:
    """Extract a spectrum for a single pixel from an AVIRIS-NG dataset."""

    radiance = ds["radiance"].sel(y=y, x=x)
    wavelengths = np.asarray(ds["wavelength_nm"].values, dtype=np.float64)
    mask = ds.get("band_mask")
    fwhm = ds.get("fwhm_nm")

    meta = {"sensor": ds.attrs.get("sensor", "avirisng")}
    if fwhm is not None:
        meta["fwhm_nm"] = np.asarray(fwhm.values, dtype=np.float64)

    return Spectrum(
        wavelengths=WavelengthGrid(wavelengths),
        values=np.asarray(radiance.values, dtype=np.float64),
        kind=SpectrumKind.RADIANCE,
        units=ds.attrs.get("units", _RAD_UNITS),
        mask=np.asarray(mask.values, dtype=bool) if mask is not None else None,
        meta=meta,
    )


def _read_file(path: Path) -> _SpectralData:
    try:
        return _read_file_hdf5(path)
    except (ModuleNotFoundError, OSError):
        return _read_file_netcdf(path)


def _read_file_hdf5(path: Path) -> _SpectralData:
    try:
        import h5py  # type: ignore[import]
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised in fallback
        raise ModuleNotFoundError("h5py is required to read AVIRIS-NG HDF5 products") from exc

    with h5py.File(path, "r") as handle:
        radiance, rad_units = _read_radiance_hdf5(handle, h5py)
        wavelengths, wavelength_units = _read_array_hdf5(handle, ["wavelength", "wavelengths"], h5py)
        if wavelengths is None:
            raise ValueError("Could not locate wavelength information in AVIRIS-NG file")
        wavelengths = np.asarray(wavelengths, dtype=np.float64)
        wavelengths_nm = _convert_wavelengths(wavelengths, wavelength_units)

        fwhm, fwhm_units = _read_array_hdf5(handle, ["fwhm", "bandwidth"], h5py)
        if fwhm is not None:
            fwhm = np.asarray(fwhm, dtype=np.float64)
            fwhm = _convert_wavelengths(fwhm, fwhm_units or wavelength_units)

        band_mask, _ = _read_array_hdf5(handle, ["band_mask", "Band_Mask", "good_bands"], h5py)
        bad_list, _ = _read_array_hdf5(handle, ["bad_band_list", "BadBands", "bad_bands"], h5py)

    radiance = _ensure_orientation(np.asarray(radiance), wavelengths_nm.shape[0])
    radiance = _convert_radiance(radiance, rad_units)

    mask = _compose_mask_arrays(band_mask, bad_list, wavelengths_nm.shape[0])
    mask = _finalize_mask(mask, wavelengths_nm)

    return _SpectralData(radiance, wavelengths_nm, fwhm, mask)


def _read_file_netcdf(path: Path) -> _SpectralData:
    data = xr.open_dataset(path).load()

    radiance = _xr_find_radiance(data)
    wavelengths_var = _xr_find_variable(data, ["wavelength_nm", "wavelength", "lambda"])
    if wavelengths_var is None:
        raise ValueError("Could not locate wavelength information in dataset")

    wavelengths_nm = _convert_wavelengths(
        np.asarray(wavelengths_var.values, dtype=np.float64),
        wavelengths_var.attrs.get("units"),
    )

    fwhm_var = _xr_find_variable(data, ["fwhm_nm", "fwhm", "bandwidth"])
    fwhm = None
    if fwhm_var is not None:
        fwhm = _convert_wavelengths(
            np.asarray(fwhm_var.values, dtype=np.float64),
            fwhm_var.attrs.get("units") or wavelengths_var.attrs.get("units"),
        )

    band_mask_var = _xr_find_variable(data, ["band_mask", "good_bands"])
    band_mask = np.asarray(band_mask_var.values) if band_mask_var is not None else None

    bad_list_var = _xr_find_variable(data, ["bad_band_list", "bad_bands"])
    bad_list = np.asarray(bad_list_var.values) if bad_list_var is not None else None

    rad_values = np.asarray(radiance.values)
    rad_values = _ensure_orientation(rad_values, wavelengths_nm.shape[0])
    rad_values = _convert_radiance(rad_values, radiance.attrs.get("units"))

    mask = _compose_mask_arrays(band_mask, bad_list, wavelengths_nm.shape[0])
    mask = _finalize_mask(mask, wavelengths_nm)

    return _SpectralData(rad_values, wavelengths_nm, fwhm, mask)


def _read_radiance_hdf5(handle, h5py_module):
    dataset = _find_dataset(handle, ["radiance", "Radiance"], h5py_module)
    if dataset is None:
        raise ValueError("Radiance dataset not found in AVIRIS-NG file")
    units = dataset.attrs.get("Units") or dataset.attrs.get("units")
    if isinstance(units, bytes):
        units = units.decode("utf-8")
    return dataset[...], units


def _read_array_hdf5(handle, names: Iterable[str], h5py_module):
    dataset = _find_dataset(handle, names, h5py_module)
    if dataset is None:
        return None, None
    units = dataset.attrs.get("Units") or dataset.attrs.get("units")
    if isinstance(units, bytes):
        units = units.decode("utf-8")
    return dataset[...], units


def _find_dataset(handle, candidates: Iterable[str], h5py_module):
    wanted = {name.lower() for name in candidates}

    def _visit(group):
        for key, item in group.items():
            if isinstance(item, h5py_module.Dataset) and key.lower() in wanted:
                return item
            if isinstance(item, h5py_module.Group):
                found = _visit(item)
                if found is not None:
                    return found
        return None

    return _visit(handle)


def _xr_find_radiance(ds: xr.Dataset) -> xr.DataArray:
    if "radiance" in ds.data_vars:
        return ds["radiance"]

    for var in ds.data_vars.values():
        if var.ndim != 3:
            continue
        dims = {d.lower() for d in var.dims}
        if {"y", "x"}.issubset(dims) or any("band" in d.lower() or "wave" in d.lower() for d in var.dims):
            return var

    raise ValueError("Could not find radiance cube in dataset")


def _xr_find_variable(ds: xr.Dataset, names: Iterable[str]) -> Optional[xr.DataArray]:
    for name in names:
        if name in ds.data_vars:
            return ds[name]
        if name in ds.coords:
            return ds.coords[name]
    return None


def _convert_wavelengths(values: np.ndarray, units: str | None) -> np.ndarray:
    if units is None:
        return np.asarray(values, dtype=np.float64)
    units_norm = units.strip().lower()
    if units_norm in {"nm", "nanometer", "nanometers"}:
        return np.asarray(values, dtype=np.float64)
    if units_norm in {"um", "micrometer", "micrometers", "micron", "microns"}:
        return np.asarray(values, dtype=np.float64) * 1_000.0
    raise ValueError(f"Unsupported wavelength units: {units}")


def _convert_radiance(values: np.ndarray, units: str | None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if units is None:
        return arr

    text = units.lower().replace("·", " ").replace("*", " ")

    prefix = 1.0
    if "microw" in text or "µw" in text or "uw" in text:
        prefix = 1e-6
    elif "milliw" in text or " mw" in text:
        prefix = 1e-3

    area = 1.0
    if any(token in text for token in ["cm^-2", "cm-2", "cm^(-2)", "per square centimeter", "cm^2"]):
        area = 1e4

    return arr * prefix * area


def _compose_mask_arrays(
    band_mask: Optional[np.ndarray], bad_list: Optional[np.ndarray], band_count: int
) -> Optional[np.ndarray]:
    mask = None

    if bad_list is not None:
        indices = np.asarray(bad_list, dtype=int).ravel()
        if indices.size:
            if np.any(indices == 0):
                bad_indices = indices
            else:
                bad_indices = indices - 1
            bad_indices = bad_indices[(bad_indices >= 0) & (bad_indices < band_count)]
            mask = np.ones(band_count, dtype=bool)
            mask[bad_indices] = False

    if band_mask is not None:
        band_mask = np.asarray(band_mask).astype(bool)
        if band_mask.shape[0] != band_count:
            raise ValueError("Band mask length mismatch")
        mask = band_mask if mask is None else (mask & band_mask)

    return mask


def _finalize_mask(mask: Optional[np.ndarray], wavelengths: np.ndarray) -> np.ndarray:
    water_mask = _water_vapor_mask(wavelengths)
    if mask is None:
        return water_mask
    mask = np.asarray(mask, dtype=bool)
    if mask.shape[0] != wavelengths.shape[0]:
        raise ValueError("Band mask length does not match wavelength vector")
    return mask & water_mask


def _water_vapor_mask(wavelengths_nm: np.ndarray) -> np.ndarray:
    mask = np.ones_like(wavelengths_nm, dtype=bool)
    for lo, hi in _WATER_VAPOR_WINDOWS_NM:
        mask &= ~((wavelengths_nm >= lo) & (wavelengths_nm <= hi))
    return mask


def _ensure_orientation(radiance: np.ndarray, band_count: int) -> np.ndarray:
    arr = np.asarray(radiance)
    if arr.ndim != 3:
        raise ValueError("Radiance cube must be 3-D")

    axes = [axis for axis, size in enumerate(arr.shape) if size == band_count]
    if not axes:
        raise ValueError("Could not determine spectral axis for radiance cube")
    spectral_axis = axes[0]
    arr = np.moveaxis(arr, spectral_axis, -1)
    return arr.astype(np.float64)

