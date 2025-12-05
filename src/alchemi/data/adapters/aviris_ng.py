"""Adapter stub for AVIRIS-NG scenes.

This module mirrors the structure of :mod:`alchemi.data.adapters.emit` but
inserts TODO markers where mission documentation is required to parse official
products.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Dict, List

import numpy as np
import xarray as xr

from alchemi.registry import srfs
from alchemi.spectral import Sample, Spectrum
from alchemi.spectral.srf import SRFMatrix as DenseSRFMatrix
from alchemi.spectral.srf import SRFProvenance
from alchemi.srf.registry import sensor_srf_from_legacy
from alchemi.srf.utils import build_gaussian_srf_matrix, default_band_widths, validate_srf_alignment

from ..io import load_avirisng_l1b

__all__ = [
    "iter_aviris_ng_pixels",
    "iter_aviris_ng_reflectance_pixels",
    "load_aviris_ng_scene",
    "load_aviris_ng_reflectance_scene",
]


_WATER_VAPOR_WINDOWS_NM: tuple[tuple[float, float], ...] = (
    (1340.0, 1460.0),
    (1790.0, 1960.0),
)


def _deep_water_vapour_mask(wavelength_nm: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(wavelength_nm, dtype=bool)
    for lo, hi in _WATER_VAPOR_WINDOWS_NM:
        mask |= (wavelength_nm >= lo) & (wavelength_nm <= hi)
    return mask


def _resolve_wavelengths(ds: xr.Dataset) -> np.ndarray:
    if "wavelength_nm" in ds:
        return np.asarray(ds["wavelength_nm"].values, dtype=np.float64)
    if "wavelength" in ds:
        wavelengths = np.asarray(ds["wavelength"].values, dtype=np.float64)
        if wavelengths.max(initial=0.0) < 100.0:
            wavelengths = wavelengths * 1_000.0
        return wavelengths
    msg = "Dataset missing wavelength coordinate"
    raise KeyError(msg)


def _quality_masks(
    band_mask: np.ndarray, *, deep_water_vapour: np.ndarray | None = None, bad_detector: np.ndarray | None = None
) -> Dict[str, np.ndarray]:
    valid = np.asarray(band_mask, dtype=bool).copy()
    if deep_water_vapour is not None:
        valid &= ~np.asarray(deep_water_vapour, dtype=bool)
    if bad_detector is not None:
        valid &= ~np.asarray(bad_detector, dtype=bool)

    quality: Dict[str, np.ndarray] = {"valid_band": valid}
    if deep_water_vapour is not None:
        quality["deep_water_vapour"] = np.asarray(deep_water_vapour, dtype=bool)
    if bad_detector is not None:
        quality["bad_detector"] = np.asarray(bad_detector, dtype=bool)
    return quality


def _srf_matrix_for_avirisng(
    wavelengths: np.ndarray, *, srf_blind: bool, fwhm: np.ndarray | None
) -> tuple[DenseSRFMatrix | None, SRFProvenance, np.ndarray]:
    widths = fwhm if fwhm is not None else default_band_widths("avirisng", wavelengths)
    if not srf_blind:
        try:
            legacy = srfs.get_srf("aviris-ng")
        except Exception:
            legacy = None
        if legacy is not None:
            sensor_srf = sensor_srf_from_legacy(
                legacy, grid=wavelengths, provenance=SRFProvenance.OFFICIAL
            )
            if sensor_srf.band_centers_nm.shape[0] == wavelengths.shape[0] and np.allclose(
                sensor_srf.band_centers_nm, wavelengths, atol=0.75
            ):
                matrix = sensor_srf.as_matrix()
                validate_srf_alignment(wavelengths, matrix.matrix, centers_nm=wavelengths)
                return matrix, sensor_srf.provenance, widths

    gaussian = build_gaussian_srf_matrix(wavelengths, widths, sensor="aviris-ng")
    return gaussian, SRFProvenance.GAUSSIAN, widths


def _band_meta(
    wavelengths: np.ndarray,
    valid_mask: np.ndarray,
    width_nm: np.ndarray,
    provenance: SRFProvenance,
) -> dict[str, np.ndarray]:
    srf_source = np.full(wavelengths.shape[0], provenance.value, dtype=object)
    return {
        "center_nm": wavelengths,
        "width_nm": width_nm,
        "valid_mask": valid_mask,
        "srf_source": srf_source,
    }


def _build_band_mask(ds: xr.Dataset, wavelengths: np.ndarray) -> np.ndarray:
    if "band_mask" in ds:
        mask = np.asarray(ds["band_mask"].values, dtype=bool)
    else:
        mask = np.ones_like(wavelengths, dtype=bool)
    return np.asarray(mask, dtype=bool)


def _build_sample(
    *,
    ds: xr.Dataset,
    path: str,
    y: int,
    x: int,
    values: np.ndarray,
    spectrum_kind: str,
    srf_blind: bool,
) -> Sample:
    wavelengths = _resolve_wavelengths(ds)
    band_mask = _build_band_mask(ds, wavelengths)
    deep_water_vapour = _deep_water_vapour_mask(wavelengths)
    band_mask = np.asarray(band_mask, dtype=bool)
    valid_mask = band_mask & ~deep_water_vapour
    fwhm = np.asarray(ds["fwhm_nm"].values, dtype=np.float64) if "fwhm_nm" in ds else None
    srf_matrix, provenance, widths = _srf_matrix_for_avirisng(wavelengths, srf_blind=srf_blind, fwhm=fwhm)

    spectrum = Spectrum(wavelength_nm=wavelengths.astype(np.float64), values=values, kind=spectrum_kind)
    ancillary = {
        "source_path": path,
        "y": int(y),
        "x": int(x),
        "srf_mode": "srf-blind" if srf_blind else "srf-aware",
        "srf_source": provenance.value,
    }

    return Sample(
        spectrum=spectrum,
        sensor_id=str(ds.attrs.get("sensor", "aviris-ng")),
        srf_matrix=srf_matrix,
        band_meta=_band_meta(wavelengths, valid_mask, widths, provenance),
        quality_masks=_quality_masks(valid_mask, deep_water_vapour=deep_water_vapour),
        ancillary=ancillary,
    )


def iter_aviris_ng_pixels(path: str, *, srf_blind: bool = False) -> Iterable[Sample]:
    """Iterate through an AVIRIS-NG cube yielding :class:`Sample` objects."""

    ds = load_avirisng_l1b(path)
    radiance = ds["radiance"]

    for y in range(radiance.shape[0]):
        for x in range(radiance.shape[1]):
            values = np.asarray(radiance[y, x, :], dtype=np.float64)
            yield _build_sample(
                ds=ds, path=path, y=y, x=x, values=values, spectrum_kind="radiance", srf_blind=srf_blind
            )


def load_aviris_ng_scene(path: str, *, srf_blind: bool = False) -> List[Sample]:
    """Materialise :func:`iter_aviris_ng_pixels` into a list."""

    return list(iter_aviris_ng_pixels(path, srf_blind=srf_blind))


def iter_aviris_ng_reflectance_pixels(path: str, *, srf_blind: bool = False) -> Iterable[Sample]:
    """Iterate through a precomputed AVIRIS-NG reflectance cube yielding Samples."""

    ds = xr.open_dataset(path).load()
    reflectance = ds["reflectance"].transpose("y", "x", "band")
    wavelengths = _resolve_wavelengths(ds)
    if "wavelength_nm" not in ds.coords:
        ds = ds.assign_coords(wavelength_nm=("band", wavelengths))

    for y in range(reflectance.shape[0]):
        for x in range(reflectance.shape[1]):
            values = np.asarray(reflectance[y, x, :], dtype=np.float64)
            yield _build_sample(
                ds=ds, path=path, y=y, x=x, values=values, spectrum_kind="reflectance", srf_blind=srf_blind
            )


def load_aviris_ng_reflectance_scene(path: str, *, srf_blind: bool = False) -> List[Sample]:
    """Materialise :func:`iter_aviris_ng_reflectance_pixels` into a list."""

    return list(iter_aviris_ng_reflectance_pixels(path, srf_blind=srf_blind))
