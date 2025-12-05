"""Adapter for EMIT scenes returning contract-compliant :class:`Sample` objects."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Dict, List

import numpy as np
import xarray as xr

from alchemi.registry import srfs
from alchemi.spectral import BandMetadata, Sample, Spectrum
from alchemi.spectral.srf import SRFMatrix as DenseSRFMatrix
from alchemi.types import QuantityKind, SRFMatrix as LegacySRFMatrix, ValueUnits
from alchemi.physics import units as qty_units
from alchemi.srf.utils import build_gaussian_srf_matrix, default_band_widths, validate_srf_alignment

from ..io.emit import WATER_VAPOR_WINDOWS_NM, load_emit_l1b
from ...io.emit_l2b import iter_high_confident_pixels, load_emit_l2b

__all__ = [
    "attach_emit_l2b_labels",
    "iter_emit_pixels",
    "iter_emit_l2a_pixels",
    "load_emit_scene",
    "load_emit_l2a_scene",
]


def _normalize_radiance_units(radiance: xr.DataArray, dataset: xr.Dataset) -> np.ndarray:
    """Convert radiance to W/m^2/sr/nm using declared metadata."""

    values = np.asarray(radiance.values, dtype=np.float64)
    raw_units = radiance.attrs.get("units") or dataset.attrs.get("radiance_units")
    src_units = qty_units.normalize_units(
        raw_units or ValueUnits.RADIANCE_W_M2_SR_NM.value, QuantityKind.RADIANCE
    )
    if src_units != ValueUnits.RADIANCE_W_M2_SR_NM:
        values = qty_units.scale_radiance_between_wavelength_units(
            values, src_units, ValueUnits.RADIANCE_W_M2_SR_NM
        )
    return values


def _ensure_wavelengths_nm(ds: xr.Dataset) -> np.ndarray:
    if "wavelength_nm" not in ds.coords:
        raise KeyError("Dataset is missing 'wavelength_nm' coordinate")
    wavelengths = np.asarray(ds["wavelength_nm"].values, dtype=np.float64)
    if wavelengths.ndim != 1:
        raise ValueError("wavelength_nm must be one-dimensional")
    if wavelengths.size > 1 and np.any(np.diff(wavelengths) <= 0):
        raise ValueError("wavelength_nm must be strictly increasing")

    coord_units = str(ds["wavelength_nm"].attrs.get("units", "nm")).lower()
    if coord_units not in {"", "nm", "nanometer", "nanometre"}:
        raise ValueError(f"wavelength_nm must be expressed in nanometres, got {coord_units!r}")
    return wavelengths


def _deep_water_vapour_mask(wavelengths_nm: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(wavelengths_nm, dtype=bool)
    for start, end in WATER_VAPOR_WINDOWS_NM:
        mask |= (wavelengths_nm >= start) & (wavelengths_nm <= end)
    return mask


def _quality_masks(
    ds: xr.Dataset, radiance: xr.DataArray, include_quality: bool, extra_band_masks: Sequence[np.ndarray] | None = None
) -> dict[str, np.ndarray]:
    wavelengths_nm = _ensure_wavelengths_nm(ds)
    valid_band = np.ones_like(wavelengths_nm, dtype=bool)

    band_masks: list[np.ndarray] = []
    if include_quality and "band_mask" in ds:
        band_masks.append(np.asarray(ds["band_mask"].values, dtype=bool))

    if extra_band_masks:
        band_masks.extend(list(extra_band_masks))

    deep_water_vapour = _deep_water_vapour_mask(wavelengths_nm)
    valid_band &= ~deep_water_vapour

    for mask in band_masks:
        valid_band &= np.asarray(mask, dtype=bool)

    quality: dict[str, np.ndarray] = {"valid_band": np.broadcast_to(valid_band, radiance.shape)}
    if np.any(deep_water_vapour):
        quality["deep_water_vapour"] = np.broadcast_to(deep_water_vapour, radiance.shape)

    if include_quality:
        for idx, mask in enumerate(band_masks):
            name = "band_mask" if idx == 0 else f"band_mask_{idx}"
            quality[name] = np.broadcast_to(np.asarray(mask, dtype=bool), radiance.shape)
    return quality


def _coerce_srf_matrix(srf_matrix: Any | None, wavelengths_nm: np.ndarray | None = None) -> Any | None:
    if srf_matrix is None:
        return None
    if hasattr(srf_matrix, "matrix"):
        return srf_matrix
    if isinstance(srf_matrix, LegacySRFMatrix):
        grid = (
            np.asarray(wavelengths_nm, dtype=np.float64)
            if wavelengths_nm is not None
            else np.unique(np.concatenate([np.asarray(b, dtype=np.float64) for b in srf_matrix.bands_nm]))
        )
        dense = np.zeros((len(srf_matrix.bands_nm), grid.shape[0]), dtype=np.float64)
        for idx, (nm, resp) in enumerate(zip(srf_matrix.bands_nm, srf_matrix.bands_resp, strict=True)):
            nm_arr = np.asarray(nm, dtype=np.float64)
            resp_arr = np.asarray(resp, dtype=np.float64)
            dense[idx] = np.interp(grid, nm_arr, resp_arr, left=0.0, right=0.0)
            area = float(np.trapezoid(dense[idx], x=grid))
            if area > 0.0:
                dense[idx] /= area
        return DenseSRFMatrix(wavelength_nm=grid, matrix=dense)
    return srf_matrix


def _resolve_emit_srf(
    wavelengths: np.ndarray,
    *,
    srf_blind: bool,
) -> tuple[DenseSRFMatrix | None, np.ndarray, str, str]:
    widths = default_band_widths("emit", wavelengths)
    srf_matrix: DenseSRFMatrix | None = None
    srf_source = "none"
    srf_mode = "srf-blind" if srf_blind else "srf-aware"

    if not srf_blind:
        try:
            raw_srf = srfs.get_srf("emit")
        except FileNotFoundError:
            raw_srf = None
        if raw_srf is not None and np.asarray(raw_srf.centers_nm, dtype=np.float64).shape[0] == wavelengths.shape[0]:
            dense = _coerce_srf_matrix(raw_srf, wavelengths_nm=wavelengths)
            if dense is not None:
                validate_srf_alignment(wavelengths, dense.matrix, centers_nm=wavelengths)
                srf_matrix = dense
                srf_source = "official"

    if srf_matrix is None and np.all(np.isfinite(widths)):
        srf_matrix = build_gaussian_srf_matrix(wavelengths, widths, sensor="emit")
        srf_source = "gaussian"

    return srf_matrix, widths, srf_source, srf_mode


def iter_emit_pixels(path: str, *, include_quality: bool = True, srf_blind: bool = False) -> Iterable[Sample]:
    """Iterate over pixels in an EMIT scene.

    Parameters
    ----------
    path:
        Path to an EMIT L1B product supported by :func:`alchemi.data.io.load_emit_l1b`.
    include_quality:
        When ``True`` include QA masks packaged alongside the dataset if present.
    """

    ds = load_emit_l1b(path)
    wavelengths = _ensure_wavelengths_nm(ds)
    radiance = ds["radiance"]
    scaled = _normalize_radiance_units(radiance, ds)

    srf_matrix, widths, srf_source, srf_mode = _resolve_emit_srf(wavelengths, srf_blind=srf_blind)
    srf_masks: list[np.ndarray] = []
    if srf_matrix is not None and getattr(srf_matrix, "bad_band_mask", None) is not None:
        srf_masks.append(~np.asarray(srf_matrix.bad_band_mask, dtype=bool))

    quality_base = _quality_masks(ds, radiance, include_quality, extra_band_masks=srf_masks)

    valid_mask = np.asarray(quality_base["valid_band"], dtype=bool)
    if valid_mask.ndim == 3:
        valid_mask = valid_mask[0, 0, :]
    band_meta = BandMetadata(
        center_nm=wavelengths,
        width_nm=widths,
        valid_mask=valid_mask,
        srf_source=np.full_like(wavelengths, srf_source, dtype=object),
    )

    for y in range(scaled.shape[0]):
        for x in range(scaled.shape[1]):
            values = scaled[y, x, :]
            spectrum = Spectrum(
                wavelength_nm=wavelengths,
                values=values,
                kind="radiance",
                units=ValueUnits.RADIANCE_W_M2_SR_NM,
            )
            quality_masks = {name: mask[y, x, :] for name, mask in quality_base.items()}
            yield Sample(
                spectrum=spectrum,
                sensor_id="emit",
                quality_masks=quality_masks,
                srf_matrix=srf_matrix,
                ancillary={
                    "source_path": path,
                    "y": int(y),
                    "x": int(x),
                    "srf_source": srf_source,
                    "srf_mode": srf_mode,
                },
                band_meta=band_meta,
            )


def load_emit_scene(path: str, *, include_quality: bool = True, srf_blind: bool = False) -> List[Sample]:
    """Load an EMIT scene into a list of :class:`Sample` objects.

    This helper materialises the iterator returned by :func:`iter_emit_pixels` for
    convenience in small benchmarks. Large scenes should prefer streaming via the
    iterator to avoid memory pressure.
    """

    return list(iter_emit_pixels(path, include_quality=include_quality, srf_blind=srf_blind))


def _load_emit_l2a(path: str) -> xr.Dataset:
    return xr.open_dataset(path)


def _normalize_reflectance_values(reflectance: xr.DataArray) -> np.ndarray:
    values = np.asarray(reflectance.values, dtype=np.float64)
    units = str(reflectance.attrs.get("units", "")).lower()
    if "%" in units or "percent" in units:
        values = values / 100.0
    if np.nanmin(values) < 0.0 or np.nanmax(values) > 1.0:
        raise ValueError("Reflectance values must lie within [0, 1]")
    return values


def iter_emit_l2a_pixels(path: str, *, include_quality: bool = True, srf_blind: bool = False) -> Iterable[Sample]:
    ds = _load_emit_l2a(path)
    wavelengths = _ensure_wavelengths_nm(ds)
    reflectance = ds.get("reflectance")
    if reflectance is None:
        reflectance = ds.get("surface_reflectance")
    if reflectance is None:
        raise KeyError("Dataset does not contain 'reflectance' variable")

    scaled = _normalize_reflectance_values(reflectance)
    srf_matrix, widths, srf_source, srf_mode = _resolve_emit_srf(wavelengths, srf_blind=srf_blind)
    srf_masks: list[np.ndarray] = []
    if srf_matrix is not None and getattr(srf_matrix, "bad_band_mask", None) is not None:
        srf_masks.append(~np.asarray(srf_matrix.bad_band_mask, dtype=bool))

    quality_base = _quality_masks(ds, reflectance, include_quality, extra_band_masks=srf_masks)
    valid_mask = np.asarray(quality_base["valid_band"], dtype=bool)
    if valid_mask.ndim == 3:
        valid_mask = valid_mask[0, 0, :]
    band_meta = BandMetadata(
        center_nm=wavelengths,
        width_nm=widths,
        valid_mask=valid_mask,
        srf_source=np.full_like(wavelengths, srf_source, dtype=object),
    )

    for y in range(scaled.shape[0]):
        for x in range(scaled.shape[1]):
            values = scaled[y, x, :]
            spectrum = Spectrum(
                wavelength_nm=wavelengths,
                values=values,
                kind="reflectance",
                units=ValueUnits.REFLECTANCE_FRACTION,
            )
            quality_masks = {name: mask[y, x, :] for name, mask in quality_base.items()}
            yield Sample(
                spectrum=spectrum,
                sensor_id="emit",
                quality_masks=quality_masks,
                srf_matrix=srf_matrix,
                ancillary={
                    "source_path": path,
                    "y": int(y),
                    "x": int(x),
                    "srf_source": srf_source,
                    "srf_mode": srf_mode,
                },
                band_meta=band_meta,
            )


def load_emit_l2a_scene(path: str, *, include_quality: bool = True, srf_blind: bool = False) -> List[Sample]:
    return list(iter_emit_l2a_pixels(path, include_quality=include_quality, srf_blind=srf_blind))


def attach_emit_l2b_labels(
    samples: Sequence[Sample],
    ds_l2b: xr.Dataset | str,
    *,
    r2_min: float = 0.9,
    mineral_var: str = "mineral_group",
    r2_var: str = "fit_r2",
) -> List[Sample]:
    if isinstance(ds_l2b, (str, bytes, np.str_)):
        ds_l2b = load_emit_l2b(str(ds_l2b))

    index = {}
    for sample in samples:
        anc = sample.ancillary
        key = (anc.get("y"), anc.get("x"))
        if None not in key:
            index[(int(key[0]), int(key[1]))] = sample

    for y_idx, x_idx, mineral_group, r2 in iter_high_confident_pixels(
        ds_l2b, r2_min=r2_min, mineral_var=mineral_var, r2_var=r2_var
    ):
        key = (y_idx, x_idx)
        sample = index.get(key)
        if sample is None:
            continue

        labels = sample.ancillary.setdefault("labels", {})
        labels["emit_l2b"] = {"mineral_group": mineral_group, "fit_r2": float(r2)}

    return list(samples)


# Legacy exports maintained for compatibility with earlier adapters. These call
# through to the new iterator implementation.
def load_emit_pixel(path: str, y: int, x: int, **_: Any) -> Spectrum:  # pragma: no cover - thin wrapper
    ds = load_emit_l1b(path)
    wavelengths = np.asarray(ds["wavelength_nm"].values, dtype=np.float64)
    radiance = ds["radiance"].isel(y=y, x=x)
    values = _normalize_radiance_units(radiance)
    return Spectrum(wavelength_nm=wavelengths, values=values, kind="radiance")
