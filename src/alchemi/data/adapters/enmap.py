"""Adapter for EnMAP products returning contract-compliant samples."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Dict, List, Sequence, Tuple
import warnings

import numpy as np
import xarray as xr

from alchemi.registry import srfs
from alchemi.registry.sensors import DEFAULT_SENSOR_REGISTRY
from alchemi.spectral import BandMetadata, Sample, Spectrum
from alchemi.spectral.srf import SRFMatrix as DenseSRFMatrix
from alchemi.types import SRFMatrix as LegacySRFMatrix, ValueUnits

from ..io import enmap as io_enmap
from ..io import enmap_pixel, load_enmap_l1b

__all__ = [
    "iter_enmap_pixels",
    "iter_enmap_l2a_pixels",
    "load_enmap_scene",
    "load_enmap_l2a_scene",
]


def _normalize_l1_dataset(path_or_pair: str | tuple[str, str]) -> xr.Dataset:
    if isinstance(path_or_pair, tuple):
        # TODO: The VNIR/SWIR pairing should follow the official delivery format.
        return load_enmap_l1b(path_or_pair[0], path_or_pair[1])

    raw = xr.load_dataset(path_or_pair)

    radiance = io_enmap._find_radiance(raw)
    radiance, spectral_dim = io_enmap._standardize_dims(radiance)
    if "wavelength_nm" in raw.coords and spectral_dim in raw["wavelength_nm"].dims:
        wavelengths = np.asarray(raw["wavelength_nm"].values, dtype=np.float64)
    else:
        wavelengths = io_enmap._extract_wavelengths(raw, spectral_dim, radiance)
    fwhm = io_enmap._extract_fwhm(raw, spectral_dim)
    band_mask = io_enmap._extract_mask(raw, spectral_dim)

    radiance = radiance.astype(np.float64)
    radiance = io_enmap._convert_radiance(radiance)

    order = np.argsort(wavelengths)
    wavelengths = wavelengths[order]
    values = np.asarray(radiance.values)[:, :, order]

    coords: Dict[str, np.ndarray] = {}
    for dim in ("y", "x"):
        if dim in radiance.coords:
            coords[dim] = np.asarray(radiance.coords[dim].values)
        else:
            idx = 0 if dim == "y" else 1
            coords[dim] = np.arange(values.shape[idx], dtype=np.int64)
    coords["band"] = np.arange(values.shape[2], dtype=np.int64)

    ds = xr.Dataset()
    ds["radiance"] = xr.DataArray(values, dims=("y", "x", "band"), coords=coords)
    ds = ds.assign_coords(wavelength_nm=("band", wavelengths.astype(np.float64)))
    if fwhm is not None:
        ds["fwhm_nm"] = ("band", np.asarray(fwhm, dtype=np.float64)[order])
    if band_mask is not None:
        ds["band_mask"] = ("band", np.asarray(band_mask, dtype=bool)[order])

    ds.attrs.update(raw.attrs)
    ds.attrs.setdefault("sensor", "enmap")
    ds.attrs.setdefault("quantity", radiance.attrs.get("quantity"))
    ds.attrs.setdefault("units", radiance.attrs.get("units"))
    return ds


def _ensure_wavelengths_nm(ds: xr.Dataset) -> np.ndarray:
    wavelengths = np.asarray(ds["wavelength_nm"].values, dtype=np.float64)
    if wavelengths.ndim != 1:
        raise ValueError("wavelength_nm must be one-dimensional")
    if wavelengths.size > 1 and np.any(np.diff(wavelengths) <= 0):
        raise ValueError("wavelength_nm must be strictly increasing")
    return wavelengths


def _normalize_radiance_units(radiance: xr.DataArray) -> np.ndarray:
    values = np.asarray(radiance.values, dtype=np.float64)
    units = str(radiance.attrs.get("units", io_enmap._RAD_UNITS)).lower()
    if "um" in units and "nm" not in units:
        values *= 1e-3
    if "mw" in units:
        values *= 1e-3
    if any(tok in units for tok in ["uw", "µw", "micro"]):
        values *= 1e-6
    return values


def _coerce_srf_matrix(raw_srf: LegacySRFMatrix | None, wavelengths_nm: np.ndarray) -> Tuple[DenseSRFMatrix | None, str]:
    if raw_srf is None:
        return None, "none"
    centers = np.asarray(raw_srf.centers_nm, dtype=np.float64)
    if centers.shape[0] != wavelengths_nm.shape[0]:
        return None, "none"
    if not np.allclose(centers, wavelengths_nm, atol=0.5):
        return None, "none"

    matrix = np.zeros((centers.shape[0], wavelengths_nm.shape[0]), dtype=np.float64)
    for idx, (nm, resp) in enumerate(zip(raw_srf.bands_nm, raw_srf.bands_resp, strict=True)):
        nm_arr = np.asarray(nm, dtype=np.float64)
        resp_arr = np.asarray(resp, dtype=np.float64)
        matrix[idx, :] = np.interp(wavelengths_nm, nm_arr, resp_arr, left=0.0, right=0.0)
        area = float(np.trapz(matrix[idx, :], x=wavelengths_nm))
        if area > 0:
            matrix[idx, :] /= area

    dense = DenseSRFMatrix(wavelength_nm=wavelengths_nm, matrix=matrix)
    return dense, "official"


def _valid_band_mask(
    wavelengths_nm: np.ndarray,
    *,
    dataset_mask: np.ndarray | None,
    srf: LegacySRFMatrix | None,
) -> np.ndarray:
    valid = np.ones_like(wavelengths_nm, dtype=bool)
    if dataset_mask is not None:
        valid &= np.asarray(dataset_mask, dtype=bool)

    if srf is not None and srf.bad_band_mask is not None:
        valid &= ~np.asarray(srf.bad_band_mask, dtype=bool)
    if srf is not None and srf.bad_band_windows_nm is not None:
        for start, end in srf.bad_band_windows_nm:
            valid &= ~((wavelengths_nm >= start) & (wavelengths_nm <= end))

    spec = DEFAULT_SENSOR_REGISTRY.get_sensor("enmap")
    if spec.absorption_windows_nm is not None:
        for start, end in spec.absorption_windows_nm:
            valid &= ~((wavelengths_nm >= start) & (wavelengths_nm <= end))
    return valid


def _quality_masks(
    base_mask: np.ndarray,
    *,
    include_quality: bool,
    band_masks: Sequence[np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    masks = [base_mask]
    if band_masks:
        masks.extend(list(band_masks))

    valid = np.ones_like(base_mask, dtype=bool)
    for mask in masks:
        valid &= np.asarray(mask, dtype=bool)

    quality: dict[str, np.ndarray] = {"valid_band": valid}
    if include_quality:
        for idx, mask in enumerate(masks):
            key = "band_mask" if idx == 0 else f"band_mask_{idx}"
            quality[key] = np.asarray(mask, dtype=bool)
    return quality


def _band_metadata(
    wavelengths_nm: np.ndarray,
    fwhm: np.ndarray | None,
    valid_mask: np.ndarray,
    *,
    srf_source: str,
) -> BandMetadata:
    return BandMetadata(
        center_nm=np.asarray(wavelengths_nm, dtype=np.float64),
        width_nm=None if fwhm is None else np.asarray(fwhm, dtype=np.float64),
        valid_mask=np.asarray(valid_mask, dtype=bool),
        srf_source=np.full_like(wavelengths_nm, srf_source, dtype=object),
    )


def iter_enmap_pixels(path_or_pair: str | tuple[str, str]) -> Iterable[Sample]:
    """Yield EnMAP L1B/L1C pixels as :class:`Sample` instances."""

    ds = _normalize_l1_dataset(path_or_pair)
    wavelengths = _ensure_wavelengths_nm(ds)
    radiance = ds["radiance"]

    try:
        raw_srf = srfs.get_srf("enmap")
    except Exception:
        raw_srf = None

    dataset_mask = np.asarray(ds["band_mask"].values, dtype=bool) if "band_mask" in ds else None
    valid = _valid_band_mask(wavelengths, dataset_mask=dataset_mask, srf=raw_srf)
    srf_matrix, srf_source = _coerce_srf_matrix(raw_srf, wavelengths)
    fwhm = np.asarray(ds["fwhm_nm"].values, dtype=np.float64) if "fwhm_nm" in ds else None

    extra_masks: list[np.ndarray] = []
    if dataset_mask is not None:
        extra_masks.append(dataset_mask)
    if raw_srf is not None and raw_srf.bad_band_mask is not None:
        extra_masks.append(~np.asarray(raw_srf.bad_band_mask, dtype=bool))
    quality_base = _quality_masks(valid, include_quality=True, band_masks=extra_masks)

    scaled = _normalize_radiance_units(radiance)

    for y in range(scaled.shape[0]):
        for x in range(scaled.shape[1]):
            values = np.asarray(scaled[y, x, :], dtype=np.float64)
            spectrum = Spectrum(
                wavelength_nm=wavelengths,
                values=values,
                kind="radiance",
                units=ValueUnits.RADIANCE_W_M2_SR_NM,
            )
            quality_masks = {name: mask.copy() for name, mask in quality_base.items()}
            band_meta = _band_metadata(wavelengths, fwhm, quality_masks["valid_band"], srf_source=srf_source)

            yield Sample(
                spectrum=spectrum,
                sensor_id="enmap",
                band_meta=band_meta,
                quality_masks=quality_masks,
                srf_matrix=srf_matrix,
                ancillary={"source_path": str(path_or_pair), "y": int(y), "x": int(x), "srf_source": srf_source},
            )


def _normalize_reflectance(reflectance: xr.DataArray) -> np.ndarray:
    values = np.asarray(reflectance.values, dtype=np.float64)
    units = str(reflectance.attrs.get("units", "")).lower()
    if "%" in units or "percent" in units:
        values = values / 100.0
    min_val = float(np.nanmin(values))
    max_val = float(np.nanmax(values))
    if min_val < 0 or max_val > 1:
        warnings.warn("Clipping reflectance values outside [0, 1]", RuntimeWarning)
        values = np.clip(values, 0.0, 1.0)
    return values


def iter_enmap_l2a_pixels(path_or_pair: str | tuple[str, str], *, include_quality: bool = True) -> Iterable[Sample]:
    ds = xr.load_dataset(path_or_pair) if not isinstance(path_or_pair, tuple) else _normalize_l1_dataset(path_or_pair)
    wavelengths = _ensure_wavelengths_nm(ds)

    reflectance = ds.get("reflectance")
    if reflectance is None:
        reflectance = ds.get("surface_reflectance")
    if reflectance is None:
        raise KeyError("Dataset does not contain 'reflectance' or 'surface_reflectance'")

    try:
        raw_srf = srfs.get_srf("enmap")
    except Exception:
        raw_srf = None

    dataset_mask = np.asarray(ds["band_mask"].values, dtype=bool) if "band_mask" in ds else None
    valid = _valid_band_mask(wavelengths, dataset_mask=dataset_mask, srf=raw_srf)
    srf_matrix, srf_source = _coerce_srf_matrix(raw_srf, wavelengths)
    fwhm = np.asarray(ds["fwhm_nm"].values, dtype=np.float64) if "fwhm_nm" in ds else None

    extra_masks: list[np.ndarray] = []
    if dataset_mask is not None:
        extra_masks.append(dataset_mask)
    if raw_srf is not None and raw_srf.bad_band_mask is not None:
        extra_masks.append(~np.asarray(raw_srf.bad_band_mask, dtype=bool))
    quality_base = _quality_masks(valid, include_quality=include_quality, band_masks=extra_masks)

    scaled = _normalize_reflectance(reflectance)

    for y in range(scaled.shape[0]):
        for x in range(scaled.shape[1]):
            values = np.asarray(scaled[y, x, :], dtype=np.float64)
            spectrum = Spectrum(
                wavelength_nm=wavelengths,
                values=values,
                kind="reflectance",
                units=ValueUnits.REFLECTANCE_FRACTION,
            )
            quality_masks = {name: mask.copy() for name, mask in quality_base.items()}
            band_meta = _band_metadata(wavelengths, fwhm, quality_masks["valid_band"], srf_source=srf_source)

            yield Sample(
                spectrum=spectrum,
                sensor_id="enmap",
                band_meta=band_meta,
                quality_masks=quality_masks,
                srf_matrix=srf_matrix,
                ancillary={"source_path": str(path_or_pair), "y": int(y), "x": int(x), "srf_source": srf_source},
            )


def load_enmap_scene(path_or_pair: str | tuple[str, str]) -> List[Sample]:
    """Load an EnMAP L1B/L1C scene into memory."""

    return list(iter_enmap_pixels(path_or_pair))


def load_enmap_l2a_scene(path_or_pair: str | tuple[str, str], *, include_quality: bool = True) -> List[Sample]:
    """Load an EnMAP L2A reflectance scene into memory."""

    return list(iter_enmap_l2a_pixels(path_or_pair, include_quality=include_quality))
