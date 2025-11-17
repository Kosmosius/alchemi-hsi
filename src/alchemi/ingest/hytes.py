"""Adapter for HyTES Level-1B brightness-temperature cubes."""

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

__all__ = ["from_hytes_bt"]


def from_hytes_bt(dataset: xr.Dataset, *, srf_id: str | None = None) -> Cube:
    """Convert a HyTES L1B brightness temperature dataset into a :class:`Cube`."""

    if "brightness_temp" not in dataset:
        raise KeyError("Dataset must contain 'brightness_temp'")
    if "wavelength_nm" not in dataset.coords:
        raise KeyError("Dataset must provide 'wavelength_nm' coordinate")

    data = np.asarray(dataset["brightness_temp"].values, dtype=np.float64)
    axis = np.asarray(dataset.coords["wavelength_nm"].values, dtype=np.float64)
    attrs = dict(dataset.attrs)

    bt_attrs = getattr(dataset["brightness_temp"], "attrs", {})
    if "units" not in attrs and "units" in bt_attrs:
        attrs["units"] = bt_attrs["units"]

    geo = geo_from_attrs(dataset.attrs)

    return Cube(
        data=data,
        axis=axis,
        axis_unit="wavelength_nm",
        value_kind="brightness_temp",
        srf_id=srf_id or attrs.get("sensor"),
        geo=geo,
        attrs=attrs,
    )
