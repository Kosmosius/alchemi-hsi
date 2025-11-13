"""Helpers for loading EMIT Level-1B radiance products."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import rasterio
import xarray as xr

from alchemi.types import Spectrum, SpectrumKind, WavelengthGrid

TARGET_RADIANCE_UNITS = "W·m⁻²·sr⁻¹·nm⁻¹"
WATER_VAPOR_WINDOWS_NM: tuple[tuple[float, float], ...] = (
    (1340.0, 1450.0),
    (1800.0, 1970.0),
    (2470.0, 2550.0),
)


def load_emit_l1b(path: str, *, band_mask: bool = True) -> xr.Dataset:
    """Load an EMIT L1B radiance cube into an :class:`xarray.Dataset`.

    Parameters
    ----------
    path:
        Path to the EMIT L1B product (GeoTIFF or HDF5 supported by Rasterio).
    band_mask:
        When ``True`` a boolean mask is generated that removes deep water-vapour
        absorption bands typically excluded from analysis.

    Returns
    -------
    xarray.Dataset
        Dataset with dimensions ``(y, x, band)`` and a ``wavelength_nm`` coordinate.
        The ``radiance`` variable is expressed in :data:`TARGET_RADIANCE_UNITS`.
    """

    with rasterio.open(path) as src:
        wavelengths_nm, source_units = _extract_metadata(src)
        data = src.read(out_dtype=np.float64)
        data *= _radiance_scale(source_units)

        coords = {
            "y": np.arange(src.height, dtype=np.int32),
            "x": np.arange(src.width, dtype=np.int32),
            "band": np.arange(src.count, dtype=np.int32),
            "wavelength_nm": ("band", wavelengths_nm),
        }

        band_selection = _compute_band_mask(wavelengths_nm, band_mask)
        ds = xr.Dataset(
            data_vars={
                "radiance": xr.DataArray(
                    np.moveaxis(data, 0, -1),
                    dims=("y", "x", "band"),
                    coords=coords,
                    attrs={"units": TARGET_RADIANCE_UNITS},
                ),
                "band_mask": xr.DataArray(
                    band_selection,
                    dims=("band",),
                    coords={"band": coords["band"]},
                ),
            },
            coords=coords,
            attrs={
                "sensor": "EMIT",
                "radiance_units": TARGET_RADIANCE_UNITS,
                "source_radiance_units": source_units or TARGET_RADIANCE_UNITS,
                "driver": src.driver,
            },
        )

    ds.coords["wavelength_nm"].attrs["units"] = "nm"
    return ds


def emit_pixel(ds: xr.Dataset, y: int, x: int) -> Spectrum:
    """Extract a single EMIT pixel as a :class:`~alchemi.types.Spectrum`."""

    if "radiance" not in ds or "wavelength_nm" not in ds.coords:
        raise KeyError("Dataset must contain 'radiance' variable and 'wavelength_nm' coordinate")

    radiance = ds["radiance"].isel(y=y, x=x)
    units = radiance.attrs.get("units") or ds.attrs.get("radiance_units") or TARGET_RADIANCE_UNITS
    values = np.asarray(radiance.values, dtype=np.float64)
    values *= _radiance_scale(units)

    mask = None
    if "band_mask" in ds:
        mask = np.asarray(ds["band_mask"].values, dtype=bool)

    wavelengths_nm = np.asarray(ds.coords["wavelength_nm"].values, dtype=np.float64)
    spectrum = Spectrum(
        wavelengths=WavelengthGrid(wavelengths_nm),
        values=values,
        kind=SpectrumKind.RADIANCE,
        units=TARGET_RADIANCE_UNITS,
        mask=mask,
        meta={
            "sensor": ds.attrs.get("sensor", "EMIT"),
            "y": int(ds.coords["y"].values[y]) if "y" in ds.coords else int(y),
            "x": int(ds.coords["x"].values[x]) if "x" in ds.coords else int(x),
        },
    )
    return spectrum


def _extract_metadata(src: rasterio.io.DatasetReader) -> tuple[np.ndarray, str | None]:
    wavelengths_nm = _extract_wavelengths(src)
    if wavelengths_nm.size != src.count:
        raise ValueError("Wavelength metadata does not match band count")

    radiance_units = _extract_radiance_units(src)
    return wavelengths_nm, radiance_units


def _extract_wavelengths(src: rasterio.io.DatasetReader) -> np.ndarray:
    tags = src.tags()
    keys = [
        "wavelength_nm",
        "wavelengths_nm",
        "wavelength",
        "wavelengths",
        "WAVELENGTH",
        "WAVELENGTHS",
    ]
    raw: np.ndarray | None = None
    for key in keys:
        if key in tags:
            raw = _parse_float_list(tags[key])
            break
    if raw is None:
        per_band: list[float] = []
        for idx in src.indexes:
            band_tags = src.tags(idx)
            value = _first_present(band_tags, keys)
            if value is None:
                raise ValueError("Missing wavelength metadata for band {idx}")
            per_band.append(float(value))
        raw = np.asarray(per_band, dtype=np.float64)

    unit_key_candidates = [
        "wavelength_units",
        "wavelength_unit",
        "WAVELENGTH_UNITS",
        "WAVELENGTH_UNIT",
    ]
    unit_value = _first_present(tags, unit_key_candidates)
    if unit_value is None:
        # Try band-level units if not provided globally
        for idx in src.indexes:
            band_tags = src.tags(idx)
            unit_value = _first_present(band_tags, unit_key_candidates)
            if unit_value:
                break

    wavelengths_nm = _ensure_nanometers(np.asarray(raw, dtype=np.float64), unit_value)
    return wavelengths_nm


def _extract_radiance_units(src: rasterio.io.DatasetReader) -> str | None:
    keys = [
        "radiance_units",
        "radiance_unit",
        "band_units",
        "units",
        "RADIANCE_UNITS",
        "RADIANCE_UNIT",
        "UNITS",
    ]
    tags = src.tags()
    value = _first_present(tags, keys)
    if value is not None:
        return value
    for idx in src.indexes:
        value = _first_present(src.tags(idx), keys)
        if value is not None:
            return value
    return None


def _compute_band_mask(wavelengths_nm: np.ndarray, enabled: bool) -> np.ndarray:
    mask = np.ones_like(wavelengths_nm, dtype=bool)
    if not enabled:
        return mask
    for lower, upper in WATER_VAPOR_WINDOWS_NM:
        mask &= ~((wavelengths_nm >= lower) & (wavelengths_nm <= upper))
    return mask


def _radiance_scale(units: str | None) -> float:
    if units is None:
        return 1.0
    normalized = _normalize_units(units)
    tokens = ("/um", "perum", "permicrom", "micrometer", "micrometre", "micron")
    if any(token in normalized for token in tokens):
        return 1.0 / 1000.0
    return 1.0


def _ensure_nanometers(values: np.ndarray, unit: str | None) -> np.ndarray:
    normalized = _normalize_units(unit) if unit is not None else None
    out = values.astype(np.float64, copy=True)
    if normalized is None:
        if np.nanmax(out) < 10.0:
            out *= 1000.0
    elif any(token in normalized for token in ("um", "microm", "micron")):
        out *= 1000.0
    elif "nm" not in normalized and "nanom" not in normalized:
        raise ValueError(f"Unsupported wavelength units: {unit}")
    if np.any(np.diff(out) <= 0):
        raise ValueError("Wavelengths must be strictly increasing")
    return out


def _parse_float_list(value: str) -> np.ndarray:
    tokens = value.replace(",", " ").split()
    return np.asarray([float(token) for token in tokens], dtype=np.float64)


def _first_present(mapping: dict[str, str], keys: Sequence[str]) -> str | None:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _normalize_units(units: str | None) -> str:
    if units is None:
        return ""
    return units.strip().lower().replace(" ", "").replace("·", ".")
