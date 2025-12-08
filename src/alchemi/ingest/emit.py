"""Adapter for EMIT Level-1B radiance cubes."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from alchemi.data.cube import Cube, GeoInfo, geo_from_attrs
from alchemi.data.validators import check_cube_health

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    import xarray as xr
else:  # pragma: no cover - optional dependency for runtime type hints
    try:
        import xarray as xr
    except ImportError:  # pragma: no cover - xarray is optional at runtime
        xr = Any  # type: ignore[assignment]

__all__ = ["from_emit_l1b"]

_LOGGER = logging.getLogger(__name__)


def _extract_geo(dataset: xr.Dataset) -> GeoInfo | None:
    geo = geo_from_attrs(getattr(dataset, "attrs", {}))
    if geo is not None:
        return geo
    rio = getattr(dataset, "rio", None)
    if rio is None:
        return None
    crs = None
    transform = None
    try:  # pragma: no cover - optional dependency
        crs = rio.crs() if callable(getattr(rio, "crs", None)) else rio.crs
    except Exception:  # pragma: no cover - optional dependency best-effort
        crs = getattr(rio, "crs", None)
    try:  # pragma: no cover - optional dependency
        transform = rio.transform() if callable(getattr(rio, "transform", None)) else rio.transform
    except Exception:  # pragma: no cover - optional dependency best-effort
        transform = getattr(rio, "transform", None)
    if crs is None and transform is None:
        return None
    return geo_from_attrs({"crs": crs, "transform": transform})


def from_emit_l1b(dataset: xr.Dataset, *, srf_id: str | None = None) -> Cube:
    """Convert an EMIT L1B :class:`xarray.Dataset` into a :class:`Cube`."""

    if "radiance" not in dataset:
        raise KeyError("Dataset must contain 'radiance'")
    if "wavelength_nm" not in dataset.coords:
        raise KeyError("Dataset must provide a 'wavelength_nm' coordinate")

    data = np.asarray(dataset["radiance"].values, dtype=np.float64)
    axis = np.asarray(dataset.coords["wavelength_nm"].values, dtype=np.float64)

    attrs = dict(dataset.attrs)
    rad_attrs = getattr(dataset["radiance"], "attrs", {})
    if "units" not in attrs and "units" in rad_attrs:
        attrs["units"] = rad_attrs["units"]

    geo = _extract_geo(dataset)

    band_mask = None
    if "band_mask" in dataset:
        band_mask = np.asarray(dataset["band_mask"].values, dtype=bool)

    cube = Cube(
        data=data,
        axis=axis,
        axis_unit="wavelength_nm",
        value_kind="radiance",
        srf_id=srf_id or attrs.get("sensor"),
        geo=geo,
        attrs=attrs,
        band_mask=band_mask,
    )
    check_cube_health(cube, logger=_LOGGER)
    return cube
