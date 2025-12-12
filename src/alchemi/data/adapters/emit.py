"""Adapter for EMIT scenes returning contract-compliant :class:`Sample` objects."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
"""EMIT adapter producing canonical :class:`~alchemi.spectral.sample.Sample` objects.

Adapter contract
----------------
* Input: EMIT L1B/L2A products following the mission schema expected by
  :mod:`alchemi.data.io.emit` and :mod:`alchemi.data.io.emit_l2b`.
* Output: per-pixel :class:`~alchemi.spectral.sample.Sample` instances whose
  :class:`~alchemi.spectral.spectrum.Spectrum` uses a nanometre grid and
  canonical units (radiance in W·m⁻²·sr⁻¹·nm⁻¹ or reflectance fraction).
* Band metadata: ``BandMetadata.center_nm`` mirrors the EMIT wavelength grid;
  ``valid_mask`` combines mission QA masks, deep-water vapour windows, and
  SRF-derived masks; ``srf_source`` records whether the SRF came from the
  registry or a Gaussian fallback.
"""

from typing import Any, Dict, List

import numpy as np
import xarray as xr

from alchemi.registry import srfs
from alchemi.spectral import BandMetadata, Sample, Spectrum, ViewingGeometry
from alchemi.spectral.sample import GeoMeta
from alchemi.spectral.srf import SRFMatrix as DenseSRFMatrix
from alchemi.types import QuantityKind, ValueUnits, WavelengthGrid
from alchemi.physics import units as qty_units
from alchemi.srf.utils import build_gaussian_srf_matrix, resolve_band_widths, validate_srf_alignment

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


def _extract_viewing_geometry(ds: xr.Dataset, y: int | None = None, x: int | None = None) -> ViewingGeometry | None:
    def _value(name: str) -> float | None:
        if name in ds:
            arr = np.asarray(ds[name].values)
            if arr.size == 1:
                return float(arr.ravel()[0])
            if y is not None and x is not None and arr.ndim >= 2:
                return float(arr[y, x])
        if name in ds.attrs:
            try:
                return float(ds.attrs[name])
            except (TypeError, ValueError):
                return None
        return None

    solar_zenith = _value("solar_zenith")
    solar_azimuth = _value("solar_azimuth")
    view_zenith = _value("view_zenith")
    view_azimuth = _value("view_azimuth")
    earth_sun_distance = _value("earth_sun_distance_au")

    if all(val is None for val in (solar_zenith, solar_azimuth, view_zenith, view_azimuth)):
        return None

    return ViewingGeometry(
        solar_zenith_deg=float(solar_zenith) if solar_zenith is not None else np.nan,
        solar_azimuth_deg=float(solar_azimuth) if solar_azimuth is not None else np.nan,
        view_zenith_deg=float(view_zenith) if view_zenith is not None else np.nan,
        view_azimuth_deg=float(view_azimuth) if view_azimuth is not None else np.nan,
        earth_sun_distance_au=None if earth_sun_distance is None else float(earth_sun_distance),
    )


def _extract_geo(ds: xr.Dataset, y: int, x: int) -> GeoMeta | None:
    def _coord(names: list[str]) -> float | None:
        for name in names:
            if name in ds.coords:
                arr = np.asarray(ds.coords[name].values)
            elif name in ds:
                arr = np.asarray(ds[name].values)
            else:
                continue
            if arr.size == 1:
                return float(arr.ravel()[0])
            if arr.ndim == 2:
                return float(arr[y, x])
        return None

    lat = _coord(["lat", "latitude"])
    lon = _coord(["lon", "longitude"])
    elev = _coord(["elev", "elevation", "height"])

    if lat is None or lon is None:
        return None
    return GeoMeta(lat=float(lat), lon=float(lon), elev=None if elev is None else float(elev))


def _deep_water_vapour_mask(wavelengths_nm: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(wavelengths_nm, dtype=bool)
    for start, end in WATER_VAPOR_WINDOWS_NM:
        mask |= (wavelengths_nm >= start) & (wavelengths_nm <= end)
    return mask


def _quality_masks(
    ds: xr.Dataset,
    radiance: xr.DataArray,
    include_quality: bool,
    extra_band_masks: Sequence[np.ndarray] | None = None,
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


def _resolve_emit_srf(
    wavelengths: np.ndarray,
    *,
    srf_blind: bool,
) -> tuple[
    DenseSRFMatrix | None,
    np.ndarray,
    str,
    str,
    np.ndarray | None,
    list[tuple[float, float]] | None,
    np.ndarray,
]:
    srf_matrix: DenseSRFMatrix | None = None
    srf_source = "none"
    srf_mode = "srf-blind" if srf_blind else "srf-aware"
    srf_bad_mask: np.ndarray | None = None
    srf_bad_windows: list[tuple[float, float]] | None = None
    sensor_srf = None

    if not srf_blind:
        try:
            sensor_srf = srfs.get_sensor_srf("emit")
        except FileNotFoundError:
            sensor_srf = None

    widths, width_from_default, _ = resolve_band_widths("emit", wavelengths, srf=sensor_srf)

    if not srf_blind and sensor_srf is not None:
        dense = sensor_srf.as_matrix()
        validate_srf_alignment(wavelengths, dense.matrix, centers_nm=wavelengths)
        srf_matrix = dense
        srf_source = sensor_srf.provenance.value
        srf_bad_mask = None if sensor_srf.valid_mask is None else ~sensor_srf.valid_mask
        raw_windows = sensor_srf.meta.get("bad_band_windows_nm") if sensor_srf.meta else None
        if raw_windows is not None:
            srf_bad_windows = [(float(lo), float(hi)) for lo, hi in raw_windows]

    if srf_matrix is None and np.all(np.isfinite(widths)):
        srf_matrix = build_gaussian_srf_matrix(wavelengths, widths, sensor="emit")
        srf_source = "gaussian"
        srf_bad_mask = None
        srf_bad_windows = None

    return srf_matrix, widths, srf_source, srf_mode, srf_bad_mask, srf_bad_windows, width_from_default


def iter_emit_pixels(
    path: str, *, include_quality: bool = True, srf_blind: bool = False
) -> Iterable[Sample]:
    """Yield per-pixel radiance :class:`Sample` objects from an EMIT L1B scene.

    The iterator walks over ``y``/``x`` and returns ``Sample`` instances with
    spectra shaped ``(L,)`` on the EMIT wavelength grid (nm). ``quality_masks``
    include ``valid_band`` plus mission masks (band_mask, water vapour, SRF QA),
    and ``band_meta`` is populated with centres, widths, and SRF provenance.
    """

    ds = load_emit_l1b(path)
    wavelengths = _ensure_wavelengths_nm(ds)
    radiance = ds["radiance"]
    scaled = _normalize_radiance_units(radiance, ds)

    srf_matrix, widths, srf_source, srf_mode, srf_bad_mask, srf_windows, width_from_default = _resolve_emit_srf(
        wavelengths, srf_blind=srf_blind
    )
    srf_masks: list[np.ndarray] = []
    if srf_bad_mask is not None:
        srf_masks.append(~np.asarray(srf_bad_mask, dtype=bool))

    quality_base = _quality_masks(ds, radiance, include_quality, extra_band_masks=srf_masks)

    if srf_bad_mask is not None:
        quality_base["srf_bad_band"] = np.broadcast_to(np.asarray(srf_bad_mask, dtype=bool), radiance.shape)
    if srf_windows is not None:
        window_mask = np.zeros_like(wavelengths, dtype=bool)
        for start, end in srf_windows:
            window_mask |= (wavelengths >= start) & (wavelengths <= end)
        quality_base["srf_bad_window"] = np.broadcast_to(window_mask, radiance.shape)
        quality_base["valid_band"] = np.asarray(quality_base["valid_band"], dtype=bool) & ~np.broadcast_to(
            window_mask, radiance.shape
        )

    valid_mask = np.asarray(quality_base["valid_band"], dtype=bool)
    if valid_mask.ndim == 3:
        valid_mask = valid_mask[0, 0, :]
    provenance = np.full_like(wavelengths, srf_source, dtype=object)
    band_meta = BandMetadata(
        center_nm=wavelengths,
        width_nm=widths,
        valid_mask=valid_mask,
        srf_source=provenance,
        srf_provenance=provenance,
        srf_approximate=np.full_like(wavelengths, srf_source != "official", dtype=bool),
        width_from_default=width_from_default,
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
            viewing_geometry = _extract_viewing_geometry(ds, y=y, x=x)
            geo = _extract_geo(ds, y, x)
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
                    **(
                        {"solar_zenith_deg": float(viewing_geometry.solar_zenith_deg)}
                        if viewing_geometry is not None and viewing_geometry.solar_zenith_deg is not None
                        else {}
                    ),
                    **(
                        {"earth_sun_distance_au": float(viewing_geometry.earth_sun_distance_au)}
                        if viewing_geometry is not None
                        and viewing_geometry.earth_sun_distance_au is not None
                        else {}
                    ),
                },
                band_meta=band_meta,
                viewing_geometry=viewing_geometry,
                geo=geo,
            )


def load_emit_scene(
    path: str, *, include_quality: bool = True, srf_blind: bool = False
) -> List[Sample]:
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


def iter_emit_l2a_pixels(
    path: str, *, include_quality: bool = True, srf_blind: bool = False
) -> Iterable[Sample]:
    """Yield per-pixel reflectance :class:`Sample` objects from an EMIT L2A scene.

    Values are TOA/surface reflectance fractions on a nanometre grid. The
    iterator mirrors :func:`iter_emit_pixels` but emits reflectance spectra and
    the same set of per-band quality masks and SRF metadata.
    """
    ds = _load_emit_l2a(path)
    wavelengths = _ensure_wavelengths_nm(ds)
    reflectance = ds.get("reflectance")
    if reflectance is None:
        reflectance = ds.get("surface_reflectance")
    if reflectance is None:
        raise KeyError("Dataset does not contain 'reflectance' variable")

    scaled = _normalize_reflectance_values(reflectance)
    srf_matrix, widths, srf_source, srf_mode, srf_bad_mask, srf_windows, width_from_default = _resolve_emit_srf(
        wavelengths, srf_blind=srf_blind
    )
    srf_masks: list[np.ndarray] = []
    if srf_bad_mask is not None:
        srf_masks.append(~np.asarray(srf_bad_mask, dtype=bool))

    quality_base = _quality_masks(ds, reflectance, include_quality, extra_band_masks=srf_masks)
    if srf_bad_mask is not None:
        quality_base["srf_bad_band"] = np.broadcast_to(np.asarray(srf_bad_mask, dtype=bool), reflectance.shape)
    if srf_windows is not None:
        window_mask = np.zeros_like(wavelengths, dtype=bool)
        for start, end in srf_windows:
            window_mask |= (wavelengths >= start) & (wavelengths <= end)
        quality_base["srf_bad_window"] = np.broadcast_to(window_mask, reflectance.shape)
        quality_base["valid_band"] = np.asarray(quality_base["valid_band"], dtype=bool) & ~np.broadcast_to(
            window_mask, reflectance.shape
        )

    valid_mask = np.asarray(quality_base["valid_band"], dtype=bool)
    if valid_mask.ndim == 3:
        valid_mask = valid_mask[0, 0, :]
    band_meta = BandMetadata(
        center_nm=wavelengths,
        width_nm=widths,
        valid_mask=valid_mask,
        srf_source=np.full_like(wavelengths, srf_source, dtype=object),
        width_from_default=width_from_default,
    )

    for y in range(scaled.shape[0]):
        for x in range(scaled.shape[1]):
            values = scaled[y, x, :]
            spectrum = Spectrum.from_surface_reflectance(
                WavelengthGrid(wavelengths),
                values,
                units=ValueUnits.REFLECTANCE_FRACTION,
            )
            quality_masks = {name: mask[y, x, :] for name, mask in quality_base.items()}
            viewing_geometry = _extract_viewing_geometry(ds, y=y, x=x)
            geo = _extract_geo(ds, y, x)
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
                    **(
                        {"solar_zenith_deg": float(viewing_geometry.solar_zenith_deg)}
                        if viewing_geometry is not None and viewing_geometry.solar_zenith_deg is not None
                        else {}
                    ),
                    **(
                        {"earth_sun_distance_au": float(viewing_geometry.earth_sun_distance_au)}
                        if viewing_geometry is not None
                        and viewing_geometry.earth_sun_distance_au is not None
                        else {}
                    ),
                },
                band_meta=band_meta,
                viewing_geometry=viewing_geometry,
                geo=geo,
            )


def load_emit_l2a_scene(
    path: str, *, include_quality: bool = True, srf_blind: bool = False
) -> List[Sample]:
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
def load_emit_pixel(
    path: str, y: int, x: int, **_: Any
) -> Spectrum:  # pragma: no cover - thin wrapper
    ds = load_emit_l1b(path)
    wavelengths = np.asarray(ds["wavelength_nm"].values, dtype=np.float64)
    radiance = ds["radiance"].isel(y=y, x=x)
    values = _normalize_radiance_units(radiance)
    return Spectrum(wavelength_nm=wavelengths, values=values, kind="radiance")
