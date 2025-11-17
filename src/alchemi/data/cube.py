"""Canonical hyperspectral cube representation and helpers."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..tokens.band_tokenizer import BandTokenizer, Tokens
from ..tokens.registry import get_default_tokenizer

__all__ = ["Cube", "GeoInfo", "geo_from_attrs"]

_ALLOWED_AXIS_UNITS = {"wavelength_nm", "wavenumber_cm1"}
_ALLOWED_VALUE_KINDS = {"radiance", "reflectance", "brightness_temp"}


@dataclass(slots=True)
class GeoInfo:
    """Geospatial metadata describing the cube layout."""

    crs: Any | None
    transform: Any | None

    def to_dict(self) -> dict[str, Any]:
        """Serialise the geospatial information into a JSON-friendly mapping."""

        return {"crs": self.crs, "transform": self.transform}


@dataclass(slots=True)
class Cube:
    """Typed wrapper around an (H, W, C) hyperspectral cube."""

    data: np.ndarray
    axis: np.ndarray
    axis_unit: str
    value_kind: str
    srf_id: str | None = None
    geo: GeoInfo | None = None
    attrs: dict[str, Any] = field(default_factory=dict)
    axis_names: tuple[str, ...] = ("y", "x", "band")
    axis_coords: Mapping[str, np.ndarray] | None = None
    band_mask: np.ndarray | None = None
    sensor: str | None = None
    units: str | None = None

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data)
        if self.data.ndim != 3:
            msg = "Cube.data must have exactly three dimensions (height, width, channels)"
            raise ValueError(msg)

        self.axis = np.asarray(self.axis, dtype=np.float64)
        if self.axis.ndim != 1:
            msg = "Cube.axis must be a one-dimensional spectral coordinate"
            raise ValueError(msg)
        if self.data.shape[-1] != self.axis.shape[0]:
            msg = "Cube.axis length must match the spectral dimension of Cube.data"
            raise ValueError(msg)

        if self.axis_unit not in _ALLOWED_AXIS_UNITS:
            msg = f"axis_unit must be one of {_ALLOWED_AXIS_UNITS!r}"
            raise ValueError(msg)

        if self.value_kind not in _ALLOWED_VALUE_KINDS:
            msg = f"value_kind must be one of {_ALLOWED_VALUE_KINDS!r}"
            raise ValueError(msg)

        if self.geo is not None and not isinstance(self.geo, GeoInfo):
            msg = "geo must be a GeoInfo instance when provided"
            raise TypeError(msg)

        if self.attrs is None:
            self.attrs = {}
        else:
            self.attrs = dict(self.attrs)

        self.axis_names = tuple(self.axis_names)
        if len(self.axis_names) != self.data.ndim:
            msg = "axis_names length must match data dimensionality"
            raise ValueError(msg)

        if self.axis_coords is not None:
            coords: dict[str, np.ndarray] = {}
            for name in self.axis_names:
                if name not in self.axis_coords:
                    continue
                coords[name] = np.asarray(self.axis_coords[name])
            self.axis_coords = coords

        if self.band_mask is not None:
            mask = np.asarray(self.band_mask, dtype=bool)
            if mask.shape != (self.band_count,):
                msg = "band_mask must align with the spectral axis"
                raise ValueError(msg)
            self.band_mask = mask

        if self.sensor is None and self.srf_id is not None:
            self.sensor = self.srf_id

        if self.units is None and "units" in self.attrs:
            self.units = str(self.attrs["units"])

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return the cube shape as ``(height, width, channels)``."""

        height, width, channels = self.data.shape
        return int(height), int(width), int(channels)

    @property
    def band_count(self) -> int:
        """Number of spectral bands represented by the cube."""

        return self.data.shape[-1]

    @property
    def values(self) -> np.ndarray:
        """Backward-compatible alias for :attr:`data`."""

        return self.data

    @property
    def axes(self) -> tuple[str, ...]:
        """Return the axis names associated with the cube."""

        return self.axis_names

    @property
    def metadata(self) -> dict[str, Any]:
        """Return metadata dictionary associated with the cube."""

        return self.attrs

    @property
    def quantity(self) -> str:
        """Legacy alias for :attr:`value_kind`."""

        return self.value_kind

    @property
    def wavelength_nm(self) -> np.ndarray | None:
        """Return spectral axis expressed in nanometres when possible."""

        if self.axis_unit == "wavelength_nm":
            return self.axis
        if self.axis_unit == "wavenumber_cm1":
            return (1.0e7 / self.axis).astype(np.float64, copy=False)
        return None

    def to_tokens(
        self,
        tokenizer: BandTokenizer | None = None,
        *,
        reducer: Callable[[np.ndarray], np.ndarray] | None = None,
        width: np.ndarray | None = None,
    ) -> Tokens:
        """Tokenise the cube's spectral axis using the provided tokenizer.

        Parameters
        ----------
        tokenizer:
            :class:`BandTokenizer` instance. When ``None`` a default preset is
            selected based on :attr:`axis_unit` and :attr:`srf_id`.
        reducer:
            Callable used to collapse the spatial dimensions into a single
            spectrum. Defaults to a ``nanmean`` across all pixels.
        width:
            Optional full-width-at-half-maximum metadata aligned with the
            spectral axis. When omitted the method attempts to source the data
            from cube attributes before delegating to tokenizer heuristics.
        """

        axis_unit = "nm" if self.axis_unit == "wavelength_nm" else "cm-1"
        sensor_id = self.srf_id or self.attrs.get("sensor")
        if sensor_id is not None:
            sensor_id = str(sensor_id)
        if tokenizer is None:
            tokenizer = get_default_tokenizer(sensor_id, axis_unit)

        if reducer is None:
            flattened = self.data.reshape(-1, self.data.shape[-1])
            values = np.nanmean(flattened, axis=0)
        else:
            values = reducer(self.data)

        values = np.asarray(values, dtype=np.float64)
        if values.shape != (self.band_count,):
            msg = "Reducer must yield a one-dimensional spectrum aligned with the cube axis"
            raise ValueError(msg)

        if width is None and self.attrs:
            for key in ("band_width", "band_width_nm", "band_fwhm", "band_fwhm_nm", "fwhm"):
                if key in self.attrs:
                    width = np.asarray(self.attrs[key])
                    break
            if width is None and "band_width_cm1" in self.attrs:
                width = np.asarray(self.attrs["band_width_cm1"])

        srf_embeddings = None
        if tokenizer.config.include_srf_embed and sensor_id:
            from ..srf.utils import build_srf_band_embeddings, load_sensor_srf

            sensor_srf = load_sensor_srf(sensor_id)
            axis_nm = self.axis if axis_unit == "nm" else 1.0e7 / self.axis
            if (
                sensor_srf is not None
                and sensor_srf.centers_nm.shape == axis_nm.shape
                and np.allclose(sensor_srf.centers_nm, axis_nm, atol=1e-3)
            ):
                srf_embeddings = build_srf_band_embeddings(sensor_srf)

        return tokenizer(
            values,
            self.axis,
            axis_unit=axis_unit,
            width=width,
            srf_row=srf_embeddings,
        )

    def to_npz(self, path: Path | str) -> None:
        """Write a compressed NumPy archive representing the cube."""

        payload: dict[str, np.ndarray] = {
            "values": self.data,
            "axis": self.axis,
        }
        if self.band_mask is not None:
            payload["band_mask"] = self.band_mask
        axis_nm = self.wavelength_nm
        if axis_nm is not None:
            payload["wavelength_nm"] = axis_nm
        payload["axis_names"] = np.asarray(self.axis_names, dtype="U32")
        if self.axis_coords:
            for name, coords in self.axis_coords.items():
                payload[f"axis_{name}"] = np.asarray(coords)
        payload["attrs_json"] = np.asarray(self._attrs_json(), dtype="U")

        np.savez_compressed(Path(path), **payload)

    def to_zarr(self, path: Path | str) -> None:
        """Write the cube to a Zarr group on disk."""

        try:
            import zarr
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("zarr is required to write canonical datasets") from exc

        root = zarr.open_group(str(Path(path)), mode="w")
        root.attrs.update(self._combined_attrs())
        root.create_dataset("values", data=self.data, overwrite=True)
        root.create_dataset("axis", data=self.axis, overwrite=True)
        if self.band_mask is not None:
            root.create_dataset("band_mask", data=self.band_mask, overwrite=True)
        axis_nm = self.wavelength_nm
        if axis_nm is not None:
            root.create_dataset("wavelength_nm", data=axis_nm, overwrite=True)

        axes_group = root.require_group("axes")
        for name, size in zip(self.axis_names, self.shape, strict=True):
            coords = self._axis_coord(name, size)
            axes_group.create_dataset(name, data=coords, overwrite=True)

    def _axis_coord(self, name: str, size: int) -> np.ndarray:
        if self.axis_coords and name in self.axis_coords:
            return np.asarray(self.axis_coords[name])
        return np.arange(size, dtype=np.int32)

    def _combined_attrs(self) -> dict[str, Any]:
        attrs = dict(self.attrs)
        if self.sensor is not None:
            attrs.setdefault("sensor", self.sensor)
        if self.units is not None:
            attrs.setdefault("units", self.units)
        attrs.setdefault("axis_unit", self.axis_unit)
        attrs.setdefault("value_kind", self.value_kind)
        attrs.setdefault("axis_names", self.axis_names)
        return attrs

    def _attrs_json(self) -> str:
        return json.dumps(self._combined_attrs(), sort_keys=True, default=str)


def geo_from_attrs(attrs: Mapping[str, Any]) -> GeoInfo | None:
    """Extract a :class:`GeoInfo` instance from dataset attributes."""

    if not attrs:
        return None

    crs = None
    transform = None

    crs_keys = ("crs", "crs_wkt", "spatial_ref", "srs")
    transform_keys = ("transform", "GeoTransform", "geotransform", "affine")

    for key in crs_keys:
        if key in attrs:
            crs = attrs[key]
            break

    for key in transform_keys:
        if key in attrs:
            transform = attrs[key]
            break

    if crs is None and transform is None:
        return None

    return GeoInfo(crs=crs, transform=transform)
