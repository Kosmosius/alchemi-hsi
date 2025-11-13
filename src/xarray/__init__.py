from __future__ import annotations

import pickle
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

__all__ = ["DataArray", "Dataset", "concat", "open_dataset"]


@dataclass(frozen=True)
class Coordinate:
    values: np.ndarray
    dims: tuple[str, ...]
    attrs: dict[str, object]

    @property
    def ndim(self) -> int:
        return self.values.ndim

    def rename(self, mapping: Mapping[str, str]) -> Coordinate:
        dims = tuple(mapping.get(d, d) for d in self.dims)
        return Coordinate(self.values, dims, dict(self.attrs))


class DataArray:
    def __init__(self, data, dims: Sequence[str], coords=None, attrs=None):
        self._data = np.asarray(data)
        self.dims = tuple(dims)
        self.attrs: dict[str, object] = dict(attrs or {})
        self.coords: dict[str, Coordinate] = {}
        if coords:
            for name, value in coords.items():
                coord = _coerce_coord(name, value)
                self.coords[name] = coord

    @property
    def values(self) -> np.ndarray:
        return self._data

    @property
    def ndim(self) -> int:
        return self._data.ndim

    def astype(self, dtype) -> DataArray:
        return DataArray(self._data.astype(dtype), self.dims, self.coords, self.attrs)

    def rename(self, mapping: Mapping[str, str]) -> DataArray:
        new_dims = tuple(mapping.get(d, d) for d in self.dims)
        new_coords = {}
        for name, coord in self.coords.items():
            new_name = mapping.get(name, name)
            new_coords[new_name] = coord.rename(mapping)
        return DataArray(self._data, new_dims, new_coords, dict(self.attrs))

    def transpose(self, *dims: str) -> DataArray:
        if not dims:
            dims = tuple(reversed(self.dims))
        order = [self.dims.index(d) for d in dims]
        data = self._data.transpose(order)
        return DataArray(data, dims, self.coords, dict(self.attrs))

    def sel(self, **indexers) -> DataArray:
        index_map = {dim: slice(None) for dim in self.dims}
        for dim, value in indexers.items():
            if dim not in self.dims:
                raise KeyError(dim)
            coord = self.coords.get(dim)
            if coord is not None:
                matches = np.where(coord.values == value)[0]
                if matches.size == 0:
                    raise KeyError(value)
                index_map[dim] = int(matches[0])
            else:
                index_map[dim] = int(value)
        slices = tuple(index_map[d] for d in self.dims)
        data = self._data[slices]
        remaining_dims = tuple(d for d in self.dims if not isinstance(index_map[d], int))
        remaining_coords: dict[str, Coordinate] = {}
        for name, coord in self.coords.items():
            if any(dim in indexers for dim in coord.dims):
                if all(dim in indexers for dim in coord.dims):
                    continue
                keep_dims = tuple(d for d in coord.dims if d not in indexers)
                if not keep_dims:
                    continue
                keep_slices = []
                for dim in coord.dims:
                    if dim in indexers:
                        matches = np.where(coord.values == indexers[dim])[0]
                        idx = int(matches[0]) if matches.size else 0
                        keep_slices.append(idx)
                    else:
                        keep_slices.append(slice(None))
                new_values = coord.values[tuple(keep_slices)]
                remaining_coords[name] = Coordinate(new_values, keep_dims, dict(coord.attrs))
            else:
                remaining_coords[name] = coord
        return DataArray(data, remaining_dims, remaining_coords, dict(self.attrs))

    def isel(self, **indexers) -> DataArray:
        selectors = []
        for dim in self.dims:
            index = indexers.get(dim, slice(None))
            if isinstance(index, slice):
                selectors.append(index)
            else:
                selectors.append(int(index))
        data = self._data[tuple(selectors)]
        remaining_dims = tuple(
            dim
            for dim, selector in zip(self.dims, selectors, strict=False)
            if not isinstance(selector, int)
        )
        remaining_coords: dict[str, Coordinate] = {}
        for name, coord in self.coords.items():
            coord_selectors = []
            for dim in coord.dims:
                coord_selectors.append(indexers.get(dim, slice(None)))
            values = coord.values[tuple(coord_selectors)]
            coord_dims = tuple(
                dim
                for dim, selector in zip(coord.dims, coord_selectors, strict=False)
                if not isinstance(selector, int)
            )
            if coord_dims:
                remaining_coords[name] = Coordinate(values, coord_dims, dict(coord.attrs))
        return DataArray(data, remaining_dims, remaining_coords, dict(self.attrs))

    def to_serializable(self):
        return {
            "data": self._data,
            "dims": self.dims,
            "attrs": dict(self.attrs),
            "coords": {name: coord for name, coord in self.coords.items()},
        }


class Dataset:
    def __init__(self, data_vars=None, coords=None):
        self.data_vars: dict[str, DataArray] = {}
        self.coords: dict[str, Coordinate] = {}
        self.attrs: dict[str, object] = {}
        if coords:
            for name, value in coords.items():
                self.coords[name] = _coerce_coord(name, value)
        if data_vars:
            for name, value in data_vars.items():
                self[name] = value

    def __getitem__(self, key: str) -> DataArray:
        if key in self.data_vars:
            return self.data_vars[key]
        if key in self.coords:
            coord = self.coords[key]
            return DataArray(coord.values, coord.dims, attrs=dict(coord.attrs))
        raise KeyError(key)

    def __setitem__(self, key: str, value) -> None:
        if isinstance(value, DataArray):
            self.data_vars[key] = value
        else:
            dims, data = value[:2]
            attrs = value[2] if len(value) > 2 else None
            coords = {}
            if isinstance(dims, str):
                dims = (dims,)
            for dim in dims:
                if dim in self.coords:
                    coords[dim] = self.coords[dim]
            self.data_vars[key] = DataArray(data, dims, coords=coords, attrs=attrs)

    def get(self, key: str):
        if key in self.data_vars:
            return self.data_vars[key]
        if key in self.coords:
            coord = self.coords[key]
            return DataArray(coord.values, coord.dims, attrs=dict(coord.attrs))
        return None

    def assign_coords(self, **coords):
        new = self.copy()
        for name, value in coords.items():
            coord = _coerce_coord(name, value)
            new.coords[name] = coord
            for arr in new.data_vars.values():
                if set(coord.dims).issubset(arr.dims):
                    arr.coords[name] = coord
        return new

    def copy(self) -> Dataset:
        new = Dataset()
        new.data_vars = {
            name: DataArray(arr.values.copy(), arr.dims, arr.coords, dict(arr.attrs))
            for name, arr in self.data_vars.items()
        }
        new.coords = {name: coord for name, coord in self.coords.items()}
        new.attrs = dict(self.attrs)
        return new

    def load(self) -> Dataset:
        return self

    @property
    def sizes(self) -> dict[str, int]:
        sizes: dict[str, int] = {}
        for arr in self.data_vars.values():
            for dim, size in zip(arr.dims, arr.values.shape, strict=False):
                sizes[dim] = size
        return sizes

    @property
    def dims(self) -> dict[str, int]:
        return self.sizes

    def to_netcdf(self, path):
        payload = {
            "data_vars": {name: arr.to_serializable() for name, arr in self.data_vars.items()},
            "coords": {name: coord for name, coord in self.coords.items()},
            "attrs": dict(self.attrs),
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)


def concat(arrays: Iterable[DataArray], dim: str) -> DataArray:
    arrays = list(arrays)
    if not arrays:
        raise ValueError("No arrays to concatenate")
    base = arrays[0]
    axis = base.dims.index(dim)
    data = np.concatenate([arr.values for arr in arrays], axis=axis)
    return DataArray(data, base.dims, base.coords, dict(base.attrs))


def open_dataset(path) -> Dataset:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    ds = Dataset()
    ds.attrs.update(payload.get("attrs", {}))
    for name, coord in payload.get("coords", {}).items():
        ds.coords[name] = coord
    for name, info in payload.get("data_vars", {}).items():
        coords = {}
        for cname, coord in info.get("coords", {}).items():
            coords[cname] = coord
        ds.data_vars[name] = DataArray(
            info["data"],
            info["dims"],
            coords=coords,
            attrs=info.get("attrs"),
        )
    return ds


def _coerce_coord(name: str, value) -> Coordinate:
    if isinstance(value, Coordinate):
        return value
    if isinstance(value, DataArray):
        return Coordinate(value.values, value.dims, dict(value.attrs))
    if isinstance(value, tuple):
        dims, data = value[:2]
        attrs = value[2] if len(value) > 2 else None
        if isinstance(dims, str):
            dims = (dims,)
        return Coordinate(np.asarray(data), tuple(dims), dict(attrs or {}))
    data = np.asarray(value)
    dims = (name,)
    return Coordinate(data, dims, {})
