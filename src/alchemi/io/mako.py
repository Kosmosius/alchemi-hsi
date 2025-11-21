"""COMEX Mako L2S radiance ingestion utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import xarray as xr

from alchemi.types import Spectrum, SpectrumKind, WavelengthGrid

__all__ = [
    "ACE_GAS_NAMES",
    "MAKO_BAND_COUNT",
    "mako_pixel_bt",
    "mako_pixel_radiance",
    "open_mako_ace",
    "open_mako_btemp",
    "open_mako_l2s",
]

MAKO_BAND_COUNT = 128
MAKO_SENSOR_ID = "mako"
_TARGET_RADIANCE_UNITS = "W·m⁻²·sr⁻¹·nm⁻¹"
_SOURCE_RADIANCE_UNITS = "µW·cm⁻²·sr⁻¹·µm⁻¹"
_MICROFLICK_TO_W_M2_SR_NM = 1e-5
_CELSIUS_TO_KELVIN = 273.15
ACE_GAS_NAMES = ["NH3", "CO2", "CH4", "NO2", "N2O"]


def open_mako_l2s(path: Path | str) -> xr.Dataset:
    """Open a COMEX Mako Level-2S radiance cube as an :class:`xarray.Dataset`.

    The returned dataset exposes the ``radiance`` variable with dimensions
    ``(y, x, band)`` and a ``wavelength_nm`` coordinate expressed in
    nanometres. Radiance values are converted from microflick
    (µW·cm⁻²·sr⁻¹·µm⁻¹) to the canonical ``W·m⁻²·sr⁻¹·nm⁻¹`` units.
    """

    data_path, header_path = _resolve_paths(Path(path))
    header = _read_envi_header(header_path)

    wavelengths_nm = _wavelengths_from_header(header)
    if wavelengths_nm.size != MAKO_BAND_COUNT:
        msg = "Mako L2S cubes must contain 128 spectral bands"
        raise ValueError(msg)

    band_mask = _band_mask_from_header(header)

    with rasterio.open(data_path) as src:
        data = src.read(out_dtype=np.float64)
        height, width = src.height, src.width

    if data.shape[0] != MAKO_BAND_COUNT:
        msg = "ENVI cube band count does not match Mako specification"
        raise ValueError(msg)

    radiance = np.moveaxis(data, 0, -1) * _MICROFLICK_TO_W_M2_SR_NM

    coords: dict[str, np.ndarray] = {
        "y": np.arange(height, dtype=np.int32),
        "x": np.arange(width, dtype=np.int32),
        "band": np.arange(MAKO_BAND_COUNT, dtype=np.int32),
    }

    ds = xr.Dataset(coords=coords)
    ds = ds.assign_coords(wavelength_nm=("band", wavelengths_nm))

    ds["radiance"] = xr.DataArray(
        radiance,
        dims=("y", "x", "band"),
        coords={**coords, "wavelength_nm": ("band", wavelengths_nm)},
        attrs={
            "units": _TARGET_RADIANCE_UNITS,
            "source_units": _SOURCE_RADIANCE_UNITS,
        },
    )

    if band_mask is not None:
        ds["band_mask"] = xr.DataArray(
            band_mask,
            dims=("band",),
            coords={"band": coords["band"]},
        )

    ds.attrs.update(
        sensor=MAKO_SENSOR_ID,
        quantity="radiance",
        radiance_units=_TARGET_RADIANCE_UNITS,
        source_radiance_units=_SOURCE_RADIANCE_UNITS,
        units=_TARGET_RADIANCE_UNITS,
    )
    ds.coords["wavelength_nm"].attrs["units"] = "nm"
    return ds


def open_mako_btemp(path: Path | str) -> xr.Dataset:
    """Open a COMEX Mako Level-3 BTEMP cube with temperatures in Kelvin."""

    data_path, header_path = _resolve_paths(Path(path))
    header = _read_envi_header(header_path)

    wavelengths_nm = _wavelengths_from_header(header)
    if wavelengths_nm.size != MAKO_BAND_COUNT:
        msg = "Mako BTEMP cubes must contain 128 spectral bands"
        raise ValueError(msg)

    band_mask = _band_mask_from_header(header)

    with rasterio.open(data_path) as src:
        data = src.read(out_dtype=np.float64)
        height, width = src.height, src.width

    if data.shape[0] != MAKO_BAND_COUNT:
        msg = "ENVI cube band count does not match Mako specification"
        raise ValueError(msg)

    bt_kelvin = np.moveaxis(data, 0, -1) + _CELSIUS_TO_KELVIN

    coords: dict[str, np.ndarray] = {
        "y": np.arange(height, dtype=np.int32),
        "x": np.arange(width, dtype=np.int32),
        "band": np.arange(MAKO_BAND_COUNT, dtype=np.int32),
    }

    ds = xr.Dataset(coords=coords)
    ds = ds.assign_coords(wavelength_nm=("band", wavelengths_nm))

    ds["bt"] = xr.DataArray(
        bt_kelvin,
        dims=("y", "x", "band"),
        coords={**coords, "wavelength_nm": ("band", wavelengths_nm)},
        attrs={"units": "K", "source_units": "°C"},
    )

    if band_mask is not None:
        ds["band_mask"] = xr.DataArray(
            band_mask,
            dims=("band",),
            coords={"band": coords["band"]},
        )

    ds.attrs.update(
        sensor=MAKO_SENSOR_ID,
        quantity="brightness_temp",
        bt_units="K",
        source_bt_units="°C",
        units="K",
    )
    ds.coords["wavelength_nm"].attrs["units"] = "nm"
    return ds


def mako_pixel_radiance(ds: xr.Dataset, row: int, col: int) -> Spectrum:
    """Extract a single pixel spectrum from a Mako L2S dataset."""

    if "radiance" not in ds.data_vars or "wavelength_nm" not in ds.coords:
        msg = "Dataset must contain 'radiance' variable and 'wavelength_nm' coordinate"
        raise KeyError(msg)

    radiance = ds["radiance"].sel(y=row, x=col)
    values = np.asarray(radiance.values, dtype=np.float64)

    units = radiance.attrs.get("units") or ds.attrs.get("radiance_units")
    if _is_microflick(units):
        values = values * _MICROFLICK_TO_W_M2_SR_NM

    wavelengths = np.asarray(ds.coords["wavelength_nm"].values, dtype=np.float64)

    mask = None
    if "band_mask" in ds.data_vars:
        mask = np.asarray(ds["band_mask"].values, dtype=bool)

    meta = {
        "sensor": ds.attrs.get("sensor", MAKO_SENSOR_ID),
        "y": int(ds.coords["y"].values[row]) if "y" in ds.coords else int(row),
        "x": int(ds.coords["x"].values[col]) if "x" in ds.coords else int(col),
    }

    return Spectrum(
        wavelengths=WavelengthGrid(wavelengths),
        values=values,
        kind=SpectrumKind.RADIANCE,
        units=_TARGET_RADIANCE_UNITS,
        mask=mask,
        meta=meta,
    )


def mako_pixel_bt(ds: xr.Dataset, row: int, col: int) -> Spectrum:
    """Extract a single pixel BTEMP spectrum with values expressed in Kelvin."""

    if "bt" not in ds.data_vars or "wavelength_nm" not in ds.coords:
        msg = "Dataset must contain 'bt' variable and 'wavelength_nm' coordinate"
        raise KeyError(msg)

    bt = ds["bt"].sel(y=row, x=col)
    values = np.asarray(bt.values, dtype=np.float64)

    units = bt.attrs.get("units") or ds.attrs.get("bt_units")
    values = _ensure_kelvin(values, units)

    wavelengths = np.asarray(ds.coords["wavelength_nm"].values, dtype=np.float64)

    mask = None
    if "band_mask" in ds.data_vars:
        mask = np.asarray(ds["band_mask"].values, dtype=bool)

    meta = {
        "sensor": ds.attrs.get("sensor", MAKO_SENSOR_ID),
        "y": int(ds.coords["y"].values[row]) if "y" in ds.coords else int(row),
        "x": int(ds.coords["x"].values[col]) if "x" in ds.coords else int(col),
    }

    return Spectrum(
        wavelengths=WavelengthGrid(wavelengths),
        values=values,
        kind=SpectrumKind.BT,
        units="K",
        mask=mask,
        meta=meta,
    )


def open_mako_ace(path: Path | str) -> xr.Dataset:
    """Open a COMEX Mako Level-3 ACE cube with named gas bands."""

    data_path, header_path = _resolve_paths(Path(path))
    _ = _read_envi_header(header_path)

    with rasterio.open(data_path) as src:
        data = src.read(out_dtype=np.float64)
        height, width = src.height, src.width

    if data.shape[0] != len(ACE_GAS_NAMES):
        msg = "Mako ACE cubes must contain five gas bands"
        raise ValueError(msg)

    ace_scores = np.moveaxis(data, 0, -1)

    coords = {
        "y": np.arange(height, dtype=np.int32),
        "x": np.arange(width, dtype=np.int32),
        "gas_band": np.arange(len(ACE_GAS_NAMES), dtype=np.int32),
        "gas_name": xr.IndexVariable("gas_band", np.asarray(ACE_GAS_NAMES, dtype=object)),
    }

    ds = xr.Dataset(coords=coords)
    ds["ace"] = xr.DataArray(
        ace_scores,
        dims=("y", "x", "gas_band"),
        coords={**coords, "gas_name": ("gas_band", ACE_GAS_NAMES)},
        attrs={"units": "dimensionless"},
    )

    ds.attrs.update(sensor=MAKO_SENSOR_ID)
    return ds


def _resolve_paths(path: Path) -> tuple[Path, Path]:
    if path.suffix.lower() == ".hdr":
        header_path = path
        data_path = path.with_suffix(".dat")
    else:
        data_path = path
        header_path = path.with_suffix(".hdr")

    if not data_path.exists():
        raise FileNotFoundError(data_path)
    if not header_path.exists():
        raise FileNotFoundError(header_path)
    return data_path, header_path


def _read_envi_header(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as stream:
        collecting: str | None = None
        buffer: list[str] = []
        for raw_line in stream:
            line = raw_line.strip()
            if not line:
                continue
            if collecting is not None:
                buffer.append(line)
                if line.endswith("}"):
                    data[collecting] = " ".join(buffer)
                    collecting = None
                    buffer = []
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip().lower()
            value = value.strip()
            if value.startswith("{") and not value.endswith("}"):
                collecting = key
                buffer = [value]
                continue
            data[key] = value
    return data


def _wavelengths_from_header(header: dict[str, str]) -> np.ndarray:
    keys = ("wavelength", "wavelengths")
    value: str | None = None
    for key in keys:
        if key in header:
            value = header[key]
            break
    if value is None:
        raise KeyError("ENVI header missing 'wavelength' definition")

    wavelengths = _parse_float_list(value)

    units = header.get("wavelength units")
    return _ensure_nanometres(wavelengths, units)


def _band_mask_from_header(header: dict[str, str]) -> np.ndarray | None:
    value = header.get("bbl")
    if value is None:
        return None
    entries = _parse_float_list(value)
    if entries.size != MAKO_BAND_COUNT:
        return None
    return entries.astype(bool)


def _parse_float_list(value: str) -> np.ndarray:
    tokens = _parse_envi_list(value)
    return np.asarray([float(token) for token in tokens], dtype=np.float64)


def _parse_envi_list(value: str) -> list[str]:
    text = value.strip()
    if text.startswith("{"):
        text = text[1:]
    if text.endswith("}"):
        text = text[:-1]
    text = text.replace("\n", " ")
    parts = [part.strip() for part in text.split(",")]
    return [part for part in parts if part]


def _ensure_nanometres(wavelengths: np.ndarray, units: str | None) -> np.ndarray:
    if units is None:
        return wavelengths * 1000.0
    normalized = units.strip().lower()
    if normalized in {"nm", "nanometer", "nanometers", "nanometre", "nanometres"}:
        return wavelengths
    if normalized in {"um", "µm", "micron", "microns", "micrometer", "micrometers"}:
        return wavelengths * 1000.0
    msg = f"Unsupported wavelength units: {units}"
    raise ValueError(msg)


def _ensure_kelvin(values: np.ndarray, units: str | None) -> np.ndarray:
    if units is None:
        return values

    normalized = units.strip().lower()
    if normalized in {"k", "kelvin"}:
        return values
    if normalized in {"c", "°c", "degc", "celsius", "degrees c", "degrees-c"}:
        return values + _CELSIUS_TO_KELVIN

    msg = f"Unsupported BT units: {units}"
    raise ValueError(msg)


def _is_microflick(units: str | None) -> bool:
    if units is None:
        return False
    tokens = units.lower().replace(" ", "")
    return "microflick" in tokens or tokens.startswith("µw/cm")
