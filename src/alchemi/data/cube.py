"""Canonical hyperspectral cube representation and helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np

from ..tokens.band_tokenizer import BandTokConfig, BandTokenizer, Tokens
from ..tokens.registry import get_default as _get_default_tokenizer

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

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return the cube shape as ``(height, width, channels)``."""

        height, width, channels = self.data.shape
        return int(height), int(width), int(channels)

    @property
    def band_count(self) -> int:
        """Number of spectral bands represented by the cube."""

        return self.data.shape[-1]

    def to_tokens(
        self,
        tokenizer: BandTokenizer | None = None,
        *,
        config: BandTokConfig | None = None,
        reducer: Callable[[np.ndarray], np.ndarray] | None = None,
        fwhm: np.ndarray | None = None,
    ) -> Tokens:
        """Tokenise the cube's spectral axis using the provided tokenizer.

        Parameters
        ----------
        tokenizer:
            :class:`BandTokenizer` instance. When ``None`` a default preset is
            selected based on :attr:`axis_unit`.
        config:
            Optional configuration override forwarded to the tokenizer.
        reducer:
            Callable used to collapse the spatial dimensions into a single
            spectrum. Defaults to a ``nanmean`` across all pixels.
        fwhm:
            Optional full-width-at-half-maximum information aligned with the
            spectral axis. When omitted the method attempts to source the data
            from cube attributes before delegating to tokenizer heuristics.
        """

        axis_unit = "nm" if self.axis_unit == "wavelength_nm" else "cm-1"
        if tokenizer is None:
            tokenizer = _get_default_tokenizer(axis_unit)

        if reducer is None:
            flattened = self.data.reshape(-1, self.data.shape[-1])
            values = np.nanmean(flattened, axis=0)
        else:
            values = reducer(self.data)

        values = np.asarray(values, dtype=np.float64)
        if values.shape != (self.band_count,):
            msg = "Reducer must yield a one-dimensional spectrum aligned with the cube axis"
            raise ValueError(msg)

        if fwhm is None and self.attrs:
            for key in ("band_fwhm", "band_fwhm_nm", "band_fwhm_cm1", "fwhm"):
                if key in self.attrs:
                    fwhm = self.attrs[key]
                    break

        return tokenizer(
            values,
            self.axis,
            axis_unit=axis_unit,
            fwhm=fwhm,
            config=config,
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
