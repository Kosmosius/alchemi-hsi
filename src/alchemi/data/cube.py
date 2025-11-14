from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

__all__ = ["Cube"]


@dataclass
class Cube:
    """Canonical in-memory representation of a hyperspectral cube."""

    sensor: str
    quantity: str
    values: np.ndarray
    axes: tuple[str, ...]
    units: str
    axis_coords: dict[str, np.ndarray] = field(default_factory=dict)
    wavelength_nm: np.ndarray | None = None
    band_mask: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_xarray(cls, dataset: xr.Dataset, *, data_var: str | None = None) -> "Cube":
        var_name = data_var or _guess_data_variable(dataset)
        data = dataset[var_name]
        values = np.asarray(data.values)
        axes = tuple(str(dim) for dim in data.dims)
        units = _guess_units(dataset, data)
        sensor = str(dataset.attrs.get("sensor", "unknown"))
        quantity = _guess_quantity(dataset, data, var_name)

        axis_coords: dict[str, np.ndarray] = {}
        for dim in axes:
            coord = data.coords.get(dim)
            if coord is not None:
                axis_coords[dim] = np.asarray(coord.values)
            else:
                axis_coords[dim] = np.arange(values.shape[data.get_axis_num(dim)])

        spectral_axis = axes[-1] if axes else None
        wavelength_nm = _extract_wavelengths(dataset, spectral_axis, values.shape[-1] if axes else 0)
        band_mask = _extract_band_mask(dataset, spectral_axis, values.shape[-1] if axes else 0)

        metadata = {
            key: _json_ready(value)
            for key, value in dataset.attrs.items()
            if key not in {"sensor", "quantity", "units"}
        }

        return cls(
            sensor=sensor,
            quantity=quantity,
            values=values,
            axes=axes,
            units=units,
            axis_coords=axis_coords,
            wavelength_nm=wavelength_nm,
            band_mask=band_mask,
            metadata=metadata,
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

    def to_npz(self, path: Path | str) -> None:
        path = Path(path)
        payload: dict[str, Any] = {
            "values": self.values,
            "axes": np.asarray(self.axes, dtype=object),
        }
        for axis, coord in self.axis_coords.items():
            payload[f"axis_{axis}"] = coord
        if self.wavelength_nm is not None:
            payload["wavelength_nm"] = self.wavelength_nm
        if self.band_mask is not None:
            payload["band_mask"] = self.band_mask

        attrs = {
            "sensor": self.sensor,
            "quantity": self.quantity,
            "units": self.units,
            "metadata": self.metadata,
        }
        payload["attrs_json"] = np.array(json.dumps(attrs, default=_json_default))
        np.savez(path, **payload)

    def to_zarr(self, path: Path | str) -> None:
        try:
            import zarr  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            msg = "zarr is required to write canonical Zarr cubes"
            raise ModuleNotFoundError(msg) from exc

        path = Path(path)
        if path.exists():
            if path.is_dir():
                for child in path.iterdir():
                    if child.is_dir():
                        _rmtree(child)
                    else:
                        child.unlink()
            else:
                path.unlink()
        store = zarr.DirectoryStore(str(path))
        root = zarr.group(store=store, overwrite=True)
        root.attrs.put({
            "sensor": self.sensor,
            "quantity": self.quantity,
            "units": self.units,
            "axes": list(self.axes),
            "metadata": _json_ready(self.metadata),
        })

        chunks = _default_chunks(self.values.shape)
        root.create_dataset("values", data=self.values, chunks=chunks, compressor=None)
        axes_group = root.create_group("axes")
        for name, coord in self.axis_coords.items():
            axes_group.create_dataset(name, data=coord, compressor=None)
        if self.wavelength_nm is not None:
            root.create_dataset("wavelength_nm", data=self.wavelength_nm, compressor=None)
        if self.band_mask is not None:
            root.create_dataset("band_mask", data=self.band_mask.astype(bool), compressor=None)

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "sensor": self.sensor,
            "quantity": self.quantity,
            "units": self.units,
            "axes": list(self.axes),
            "values": self.values,
            "axis_coords": {k: v for k, v in self.axis_coords.items()},
        }
        if self.wavelength_nm is not None:
            data["wavelength_nm"] = self.wavelength_nm
        if self.band_mask is not None:
            data["band_mask"] = self.band_mask
        if self.metadata:
            data["metadata"] = self.metadata
        return data


def _guess_data_variable(dataset: xr.Dataset) -> str:
    if dataset.attrs.get("quantity") and dataset.attrs["quantity"] in dataset:
        return str(dataset.attrs["quantity"])
    priority = (
        "radiance",
        "reflectance",
        "brightness_temp",
        "brightness_temperature",
        "bt",
    )
    for name in priority:
        if name in dataset.data_vars:
            return name
    for name, var in dataset.data_vars.items():
        if var.ndim == 3:
            return name
    if dataset.data_vars:
        return next(iter(dataset.data_vars))
    raise ValueError("Dataset contains no variables")


def _guess_units(dataset: xr.Dataset, data: xr.DataArray) -> str:
    units = data.attrs.get("units")
    if units:
        return str(units)
    for key in ("units", "radiance_units", "brightness_temp_units", "bt_units"):
        value = dataset.attrs.get(key)
        if value:
            return str(value)
    return ""


def _guess_quantity(dataset: xr.Dataset, data: xr.DataArray, default: str) -> str:
    quantity = data.attrs.get("quantity") or dataset.attrs.get("quantity")
    if quantity:
        return str(quantity)
    return default


def _extract_wavelengths(dataset: xr.Dataset, spectral_axis: str | None, bands: int) -> np.ndarray | None:
    if "wavelength_nm" in dataset.coords:
        coord = dataset.coords["wavelength_nm"]
        if coord.ndim == 1 and (spectral_axis is None or spectral_axis in coord.dims):
            arr = np.asarray(coord.values, dtype=np.float64)
            if arr.size == bands:
                return arr
    for key in ("wavelength", "lambda", "wavelengths"):
        if key in dataset.coords:
            coord = dataset.coords[key]
            if coord.ndim == 1 and (spectral_axis is None or spectral_axis in coord.dims):
                arr = np.asarray(coord.values, dtype=np.float64)
                if arr.size == bands:
                    return arr
    return None


def _extract_band_mask(dataset: xr.Dataset, spectral_axis: str | None, bands: int) -> np.ndarray | None:
    if "band_mask" in dataset.data_vars:
        mask = dataset["band_mask"]
        if mask.ndim == 1 and (spectral_axis is None or spectral_axis in mask.dims):
            arr = np.asarray(mask.values, dtype=bool)
            if arr.size == bands:
                return arr
    return None


def _default_chunks(shape: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(max(1, min(64, dim)) for dim in shape)


def _json_ready(value: Any) -> Any:
    return json.loads(json.dumps(value, default=_json_default))


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _rmtree(path: Path) -> None:
    for child in path.iterdir():
        if child.is_dir():
            _rmtree(child)
        else:
            child.unlink()
    path.rmdir()
