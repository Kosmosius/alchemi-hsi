"""Adapters for COMEX Mako radiance and brightness-temperature cubes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from alchemi.data.cube import Cube, geo_from_attrs

if TYPE_CHECKING:  # pragma: no cover
    import xarray as xr
else:  # pragma: no cover - optional dependency for runtime type hints
    try:
        import xarray as xr
    except ImportError:  # pragma: no cover - xarray is optional at runtime
        xr = Any  # type: ignore[assignment]

__all__ = ["from_mako_l2s", "from_mako_l3"]


def _prepare(
    dataset: xr.Dataset, variable: str
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if variable not in dataset:
        raise KeyError(f"Dataset must contain '{variable}'")
    if "wavelength_nm" not in dataset.coords:
        raise KeyError("Dataset must provide 'wavelength_nm' coordinate")

    data = np.asarray(dataset[variable].values, dtype=np.float64)
    axis = np.asarray(dataset.coords["wavelength_nm"].values, dtype=np.float64)
    attrs = dict(dataset.attrs)

    var_attrs = getattr(dataset[variable], "attrs", {})
    if "units" not in attrs and "units" in var_attrs:
        attrs["units"] = var_attrs["units"]

    return data, axis, attrs


def from_mako_l2s(dataset: xr.Dataset, *, srf_id: str | None = None) -> Cube:
    """Convert a Mako L2S radiance dataset into a :class:`Cube`."""

    data, axis, attrs = _prepare(dataset, "radiance")

    return Cube(
        data=data,
        axis=axis,
        axis_unit="wavelength_nm",
        value_kind="radiance",
        srf_id=srf_id or attrs.get("sensor"),
        geo=geo_from_attrs(dataset.attrs),
        attrs=attrs,
    )


def from_mako_l3(dataset: xr.Dataset, *, srf_id: str | None = None) -> Cube:
    """Convert a Mako Level-3 BTEMP dataset into a :class:`Cube`."""

    data, axis, attrs = _prepare(dataset, "bt")

    return Cube(
        data=data,
        axis=axis,
        axis_unit="wavelength_nm",
        value_kind="brightness_temp",
        srf_id=srf_id or attrs.get("sensor"),
        geo=geo_from_attrs(dataset.attrs),
        attrs=attrs,
    )
