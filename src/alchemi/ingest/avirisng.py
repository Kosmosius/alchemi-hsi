"""Adapter for AVIRIS-NG Level-1B radiance cubes."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from alchemi.data.cube import Cube, geo_from_attrs
from alchemi.data.validators import check_cube_health

if TYPE_CHECKING:  # pragma: no cover
    import xarray as xr
else:  # pragma: no cover - optional dependency for runtime type hints
    try:
        import xarray as xr
    except ImportError:  # pragma: no cover - xarray is optional at runtime
        xr = Any  # type: ignore[assignment]

__all__ = ["from_avirisng_l1b"]

_LOGGER = logging.getLogger(__name__)


def from_avirisng_l1b(dataset: xr.Dataset, *, srf_id: str | None = None) -> Cube:
    """Convert an AVIRIS-NG L1B radiance dataset into a :class:`Cube`."""

    if "radiance" not in dataset:
        raise KeyError("Dataset must contain 'radiance'")
    if "wavelength_nm" not in dataset.coords:
        raise KeyError("Dataset must provide 'wavelength_nm' coordinate")

    data = np.asarray(dataset["radiance"].values, dtype=np.float64)
    axis = np.asarray(dataset.coords["wavelength_nm"].values, dtype=np.float64)
    attrs = dict(dataset.attrs)

    rad_attrs = getattr(dataset["radiance"], "attrs", {})
    if "units" not in attrs and "units" in rad_attrs:
        attrs["units"] = rad_attrs["units"]

    geo = geo_from_attrs(dataset.attrs)

    band_mask = None
    if "band_mask" in dataset:
        band_mask = np.asarray(dataset["band_mask"].values, dtype=bool)

    sensor_id = srf_id or attrs.get("sensor")
    if sensor_id is not None and str(sensor_id).lower() == "avirisng":
        sensor_id = "aviris-ng"

    cube = Cube(
        data=data,
        axis=axis,
        axis_unit="wavelength_nm",
        value_kind="radiance",
        srf_id=sensor_id,
        geo=geo,
        attrs=attrs,
        band_mask=band_mask,
    )
    check_cube_health(cube, logger=_LOGGER)
    return cube
