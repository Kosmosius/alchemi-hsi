"""Adapter for AVIRIS-NG Level-1B radiance cubes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from alchemi.data.cube import Cube, geo_from_attrs

if TYPE_CHECKING:  # pragma: no cover
    import xarray as xr

__all__ = ["from_avirisng_l1b"]


def from_avirisng_l1b(dataset: "xr.Dataset", *, srf_id: str | None = None) -> Cube:
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

    return Cube(
        data=data,
        axis=axis,
        axis_unit="wavelength_nm",
        value_kind="radiance",
        srf_id=srf_id or attrs.get("sensor"),
        geo=geo,
        attrs=attrs,
    )
