"""Canonical hyperspectral cube representation and helpers.

This module defines the :class:`Cube` type that holds spatial hyperspectral
data. Individual pixel spectra can be extracted as
:class:`alchemi.spectral.Sample` instances via :meth:`Cube.sample_at`,
preserving wavelength grids, quantity kinds, and relevant sensor metadata.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from alchemi.spectral import BandMetadata, Sample, Spectrum
from alchemi.wavelengths import check_monotonic, to_nm
from alchemi.tokens.band_tokenizer import AxisUnit, BandTokenizer, Tokens
from alchemi.tokens.registry import get_default_tokenizer
from alchemi.types import QuantityKind, ValueUnits, WavelengthGrid, _normalize_quantity_kind, _normalize_value_units

__all__ = ["Cube", "GeoInfo", "geo_from_attrs"]

_ALLOWED_AXIS_UNITS = {"wavelength_nm", "wavenumber_cm1"}
_ALLOWED_VALUE_KINDS = {QuantityKind.RADIANCE, QuantityKind.REFLECTANCE, QuantityKind.BRIGHTNESS_T}


@dataclass(slots=True)
class GeoInfo:
    """Geospatial metadata describing the cube layout."""

    crs: Any | None
    transform: Any | None

    def to_dict(self) -> dict[str, Any]:
        """Serialise the geospatial information into a JSON-friendly mapping."""

        return {"crs": self.crs, "transform": self.transform}


@dataclass(slots=True, init=False)
class Cube:
    """Typed wrapper around an (H, W, C) hyperspectral cube.

    A ``Cube`` carries a spectral axis (``axis`` / ``axis_unit``), a quantity
    kind (``value_kind``), and optional sensor / unit metadata. When converting
    individual pixels to :class:`alchemi.types.Sample` objects these fields must
    be preserved so that downstream lab-style processing sees a consistent
    wavelength grid and :class:`alchemi.types.SpectrumKind`.
    """

    data: np.ndarray
    axis: np.ndarray
    axis_unit: str
    value_kind: QuantityKind
    srf_id: str | None = None
    geo: GeoInfo | None = None
    attrs: dict[str, Any] = field(default_factory=dict)
    axis_names: tuple[str, ...] = ("y", "x", "band")
    axis_coords: Mapping[str, np.ndarray] | None = None
    band_mask: np.ndarray | None = None
    sensor: str | None = None
    units: ValueUnits | None = None

    def __init__(
        self,
        data: np.ndarray | None = None,
        axis: np.ndarray | None = None,
        axis_unit: str | None = None,
        value_kind: QuantityKind | str | None = None,
        *,
        srf_id: str | None = None,
        geo: GeoInfo | None = None,
        attrs: Mapping[str, Any] | None = None,
        **legacy: Any,
    ) -> None:
        # Legacy argument handling
        if data is None:
            if "values" in legacy:
                data = np.asarray(legacy.pop("values"))
            else:
                raise TypeError("Cube() missing required argument 'data'")
        if axis is None:
            if "wavelength_nm" in legacy:
                axis = np.asarray(legacy.pop("wavelength_nm"))
                axis_unit = axis_unit or "wavelength_nm"
            else:
                raise TypeError("Cube() missing required argument 'axis'")
        axis_unit = axis_unit or "wavelength_nm"
        if value_kind is None:
            value_kind = legacy.pop("quantity", "radiance")

        # Merge metadata sources
        metadata: dict[str, Any] = dict(attrs or {})
        meta = legacy.pop("metadata", None)
        if meta:
            metadata.update(meta)
        # Lift common legacy keys into metadata
        for key in ("sensor", "units", "axes", "axis_coords", "band_mask"):
            if key in legacy:
                metadata[key] = legacy.pop(key)
        metadata.update(legacy)

        self.data = data
        self.axis = axis
        self.axis_unit = axis_unit
        self.value_kind = _normalize_quantity_kind(value_kind)
        self.srf_id = srf_id
        self.geo = geo
        self.attrs = metadata
        # Defaults for extra fields
        self.axis_names = ("y", "x", "band")
        self.axis_coords = None
        self.band_mask = None
        self.sensor = None
        self.units = None

        self.__post_init__()

    def __post_init__(self) -> None:
        # Normalise data and axis
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
            msg = f"value_kind must be one of {[kind.value for kind in _ALLOWED_VALUE_KINDS]!r}"
            raise ValueError(msg)

        if self.geo is not None and not isinstance(self.geo, GeoInfo):
            msg = "geo must be a GeoInfo instance when provided"
            raise TypeError(msg)

        # Validate spectral axis monotonicity in nm
        check_monotonic(self._axis_wavelength_nm())

        # Normalise attrs
        self.attrs = dict(self.attrs or {})

        # Axis names: prefer explicit metadata if present
        axes_meta = self.attrs.get("axes")
        if axes_meta is not None:
            self.axis_names = tuple(axes_meta)
        else:
            self.axis_names = tuple(self.axis_names)
        if len(self.axis_names) != self.data.ndim:
            msg = "axis_names length must match data dimensionality"
            raise ValueError(msg)

        # Axis coords: merge explicit field and metadata
        coords_src: Mapping[str, Any] | None = self.axis_coords or self.attrs.get("axis_coords")
        if coords_src is not None:
            coords: dict[str, np.ndarray] = {}
            for name in self.axis_names:
                if name not in coords_src:
                    continue
                coords[name] = np.asarray(coords_src[name])
            self.axis_coords = coords
        else:
            self.axis_coords = None

        # Band mask: prefer explicit field, but fall back to metadata
        mask = self.band_mask
        if mask is None and "band_mask" in self.attrs:
            mask = self.attrs["band_mask"]
        if mask is not None:
            mask_arr = np.asarray(mask, dtype=bool)
            if mask_arr.shape != (self.band_count,):
                msg = "band_mask must align with the spectral axis"
                raise ValueError(msg)
            self.band_mask = mask_arr
            self.attrs["band_mask"] = mask_arr  # keep in sync with metadata

        # Sensor / units: derive from metadata / SRF when not provided
        if self.sensor is None:
            sensor_meta = self.attrs.get("sensor")
            if sensor_meta is not None:
                self.sensor = str(sensor_meta)
            elif self.srf_id is not None:
                self.sensor = self.srf_id

        if self.units is None and "units" in self.attrs:
            self.units = _normalize_value_units(self.attrs["units"], self.value_kind)

    # ---------- Basic properties ----------

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return the cube shape as ``(height, width, channels)``."""

        height, width, channels = self.data.shape
        return int(height), int(width), int(channels)

    @property
    def values(self) -> np.ndarray:
        """Backward-compatible alias for :attr:`data`."""

        return self.data

    @property
    def band_count(self) -> int:
        """Number of spectral bands represented by the cube."""

        return int(self.data.shape[-1])

    def _axis_wavelength_nm(self) -> np.ndarray:
        """Return the spectral axis expressed in nanometres.

        ``Sample`` uses :class:`~alchemi.types.WavelengthGrid`, which is always
        expressed in nanometres. When the cube is parameterised in wavenumbers
        the axis is converted on the fly.
        """

        if self.axis_unit == "wavelength_nm":
            return to_nm(self.axis, "nm")
        if self.axis_unit == "wavenumber_cm1":
            return to_nm(self.axis, "cm-1")

        msg = f"Cannot convert axis_unit {self.axis_unit!r} to wavelengths"
        raise ValueError(msg)

    def sample_at(self, row: int, col: int) -> Sample:
        """Extract a canonical :class:`alchemi.spectral.Sample` for ``(row, col)``.

        The returned sample shares the cube's spectral axis (converted to
        nanometres) and value kind. Sensor identifiers are sourced from
        :attr:`sensor`, ``srf_id``, or ``attrs['sensor']`` in that order. Band
        masks (if present) are propagated into band metadata and quality masks.
        """

        height, width, _ = self.shape
        if not (0 <= row < height and 0 <= col < width):
            raise IndexError(
                f"Pixel indices out of bounds for cube shape {self.shape}: {(row, col)}"
            )

        spectrum_kind = self.value_kind
        wavelengths_nm = self._axis_wavelength_nm().copy()
        values = np.asarray(self.data[row, col, :], dtype=np.float64)
        sensor_id = self.sensor or self.srf_id or self.attrs.get("sensor") or "unknown"

        extras = {k: v for k, v in self.attrs.items() if k not in {"sensor"}}
        valid_mask = (
            np.asarray(self.band_mask, dtype=bool) if self.band_mask is not None else np.ones_like(wavelengths_nm, dtype=bool)
        )
        band_meta = BandMetadata(
            center_nm=wavelengths_nm,
            width_nm=np.full_like(wavelengths_nm, np.nan, dtype=float),
            valid_mask=valid_mask,
            srf_source=np.array(["unknown"] * wavelengths_nm.shape[0], dtype=object),
        )

        quality_masks: dict[str, np.ndarray] = {}
        if self.band_mask is not None:
            quality_masks["band_mask"] = valid_mask

        spectrum = Spectrum(
            wavelength_nm=wavelengths_nm,
            values=values,
            kind=spectrum_kind.value,
            mask=valid_mask,
        )

        return Sample(
            spectrum=spectrum,
            sensor_id=str(sensor_id),
            band_meta=band_meta,
            quality_masks=quality_masks,
            ancillary={"row": int(row), "col": int(col), "cube_attrs": extras},
        )

    @property
    def axes(self) -> tuple[str, ...]:
        """Return the axis names associated with the cube.

        Prefers metadata-specified axes when present, otherwise uses
        :attr:`axis_names`.
        """

        axes = self.attrs.get("axes")
        if axes is not None:
            return tuple(axes)
        return self.axis_names

    @property
    def metadata(self) -> dict[str, Any]:
        """Return metadata dictionary associated with the cube."""

        return self.attrs

    @property
    def quantity(self) -> str:
        """Legacy alias for :attr:`value_kind`."""

        return self.value_kind.value

    @property
    def wavelength_nm(self) -> np.ndarray | None:
        """Return spectral axis expressed in nanometres when possible."""

        if self.axis_unit == "wavelength_nm":
            return self.axis
        if self.axis_unit == "wavenumber_cm1":
            converted = (1.0e7 / self.axis).astype(np.float64, copy=False)
            return converted
        return None

    def iter_tiles(
        self,
        tile_h: int,
        tile_w: int,
        step_h: int | None = None,
        step_w: int | None = None,
    ) -> Iterator[tuple[slice, slice, Cube]]:
        """Iterate over spatial tiles of the cube.

        Parameters
        ----------
        tile_h, tile_w:
            Height and width of each tile.
        step_h, step_w:
            Stride between tile starts. Defaults to non-overlapping tiles
            (``tile_h``/``tile_w``).

        Yields
        ------
        tuple
            ``(row_slice, col_slice, subcube)`` for each tile. The ``subcube``
            shares the underlying data array where possible.
        """

        if tile_h <= 0 or tile_w <= 0:
            msg = "tile_h and tile_w must be positive"
            raise ValueError(msg)

        step_h = tile_h if step_h is None else step_h
        step_w = tile_w if step_w is None else step_w
        if step_h <= 0 or step_w <= 0:
            msg = "step_h and step_w must be positive"
            raise ValueError(msg)

        height, width, _ = self.shape

        for row_start in range(0, height, step_h):
            row_end = min(row_start + tile_h, height)
            row_slice = slice(row_start, row_end)
            for col_start in range(0, width, step_w):
                col_end = min(col_start + tile_w, width)
                col_slice = slice(col_start, col_end)

                metadata = dict(self.attrs)
                if self.sensor is not None:
                    metadata.setdefault("sensor", self.sensor)
                if self.units is not None:
                    metadata.setdefault("units", self.units)
                metadata.setdefault("axes", self.axis_names)
                if self.axis_coords is not None:
                    metadata.setdefault("axis_coords", self.axis_coords)

                subcube = Cube(
                    data=self.data[row_slice, col_slice, :],
                    axis=self.axis,
                    axis_unit=self.axis_unit,
                    value_kind=self.value_kind,
                    srf_id=self.srf_id,
                    geo=self.geo,
                    attrs=metadata,
                )
                yield row_slice, col_slice, subcube

    # ---------- Tokenisation ----------

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
            :class:`BandTokenizer` instance. When ``None`` a preset from
            :func:`alchemi.tokens.registry.get_default_tokenizer` is selected
            based on the cube's resolved sensor id (``srf_id`` / ``sensor``) and
            axis units (``axis_unit`` translated to ``"nm"`` or ``"cm-1"``).
        reducer:
            Callable used to collapse the spatial dimensions into a single
            spectrum. Defaults to a ``nanmean`` across all pixels.
        width:
            Optional full-width-at-half-maximum metadata aligned with the
            spectral axis. When omitted the method attempts to source the data
            from cube attributes before delegating to tokenizer heuristics (for
            example, synthetic widths via :func:`alchemi.srf.utils.default_band_widths`).

        Notes
        -----
        If the chosen tokenizer includes SRF embeddings, SRF features are
        lazily loaded using the resolved sensor id. When the SRF centres match
        the cube axis (or its nanometre conversion) within tolerance,
        ``srf_row`` is passed to the tokenizer; otherwise SRF data is ignored.
        """

        axis_unit: AxisUnit = "nm" if self.axis_unit == "wavelength_nm" else "cm-1"
        sensor_id = self.srf_id or self.sensor or self.attrs.get("sensor")
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

    # ---------- Canonical serialisation ----------

    def to_npz(self, path: Path | str) -> None:
        """Write a compressed NumPy archive representing the cube.

        Includes legacy fields (axis_unit, value_kind, attrs_json) as well as
        richer axis / mask metadata.
        """

        payload: dict[str, Any] = {
            "values": self.data,
            "axis": self.axis,
            # Legacy scalar metadata as arrays
            "axis_unit": np.asarray(self.axis_unit),
            "value_kind": np.asarray(self.value_kind.value),
        }

        # Optional band mask
        if self.band_mask is not None:
            payload["band_mask"] = self.band_mask

        # Optional wavelength in nm
        axis_nm = self.wavelength_nm
        if axis_nm is not None:
            payload["wavelength_nm"] = axis_nm

        # Axis names / coords for richer IO
        payload["axis_names"] = np.asarray(self.axis_names, dtype="U32")
        if self.axis_coords:
            for name, coords in self.axis_coords.items():
                payload[f"axis_{name}"] = np.asarray(coords)

        # Combined attributes as JSON string
        payload["attrs_json"] = np.asarray(self._attrs_json(), dtype="U")

        np.savez_compressed(Path(path), **payload)

    def to_zarr(self, path: Path | str) -> None:
        """Write the cube to a Zarr group on disk.

        Stores both the richer canonical layout (axes group, band_mask,
        wavelength_nm, combined attrs) and the older axis_unit/value_kind/
        metadata JSON attributes for backward compatibility.
        """

        try:
            import zarr  # pragma: no cover - optional dependency
        except ModuleNotFoundError as exc:  # pragma: no cover - guard
            raise RuntimeError("zarr is required to write canonical datasets") from exc

        root = zarr.open_group(str(Path(path)), mode="w")

        # Rich combined attributes
        combined = self._combined_attrs()
        root.attrs.update(combined)

        # Backwards-compatible metadata blob
        root.attrs["axis_unit"] = self.axis_unit
        root.attrs["value_kind"] = self.value_kind.value
        root.attrs["metadata"] = _encode_metadata(self.attrs)

        # Core datasets
        root.create_dataset("values", data=self.data, overwrite=True)
        root.create_dataset("axis", data=self.axis, overwrite=True)
        if self.band_mask is not None:
            root.create_dataset("band_mask", data=self.band_mask, overwrite=True)

        axis_nm = self.wavelength_nm
        if axis_nm is not None:
            root.create_dataset("wavelength_nm", data=axis_nm, overwrite=True)

        # Axis coordinates
        axes_group = root.require_group("axes")
        for name, size in zip(self.axis_names, self.shape, strict=True):
            coords = self._axis_coord(name, size)
            axes_group.create_dataset(name, data=coords, overwrite=True)

    def _axis_coord(self, name: str, size: int) -> np.ndarray:
        if self.axis_coords and name in self.axis_coords:
            return np.asarray(self.axis_coords[name])
        return np.arange(size, dtype=np.int32)

    def _combined_attrs(self) -> dict[str, Any]:
        """Attributes used for canonical serialisation."""

        attrs = dict(self.attrs)
        if self.sensor is not None:
            attrs.setdefault("sensor", self.sensor)
        if self.units is not None:
            attrs.setdefault("units", self.units.value)
        attrs.setdefault("axis_unit", self.axis_unit)
        attrs.setdefault("value_kind", self.value_kind.value)
        attrs.setdefault("axis_names", self.axis_names)
        if self.axis_coords is not None:
            attrs.setdefault("axis_coords", self.axis_coords)
        return attrs

    def _attrs_json(self) -> str:
        # Reuse the shared metadata encoder for the combined attrs
        return _encode_metadata(self._combined_attrs())

    # ---------- Construction from xarray ----------

    @classmethod
    def from_xarray(cls, dataset: Any) -> Cube:
        """Create a Cube from an xarray Dataset.

        Preserves axis names / coords in metadata, and chooses an axis_unit
        based on the presence of a 'band' coordinate.
        """
        data_var = next(iter(dataset.data_vars))
        data_arr = dataset[data_var]
        data = np.asarray(data_arr.values)

        axes = tuple(data_arr.dims)
        axis_coords = {
            name: np.asarray(dataset.coords[name].values) for name in axes if name in dataset.coords
        }
        spectral = axis_coords.get("band")
        axis = spectral if spectral is not None else np.arange(data.shape[-1], dtype=np.float64)
        axis_unit = "wavelength_nm" if spectral is not None else "wavenumber_cm1"

        attrs = dict(dataset.attrs)
        attrs.setdefault("axes", axes)
        attrs.setdefault("axis_coords", axis_coords)

        return cls(
            data=data,
            axis=axis,
            axis_unit=axis_unit,
            value_kind=attrs.get("quantity", "radiance"),
            attrs=attrs,
        )


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


def _encode_metadata(attrs: Mapping[str, Any]) -> str:
    def _convert(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.generic,)):
            return value.item()
        if isinstance(value, dict):
            return {k: _convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_convert(v) for v in value]
        return value

    return json.dumps({k: _convert(v) for k, v in attrs.items()})
