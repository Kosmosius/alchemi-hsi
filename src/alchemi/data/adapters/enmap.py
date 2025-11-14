"""Adapter utilities for EnMAP samples."""

from __future__ import annotations

import xarray as xr
from alchemi.types import Sample, SampleMeta
from alchemi.data.io import enmap_pixel

__all__ = ["load_enmap_pixel"]


def load_enmap_pixel(ds: xr.Dataset, position: tuple[int, int]) -> Sample:
    """Create a :class:`~alchemi.types.Sample` from a dataset and (y, x) index."""

    spectrum = enmap_pixel(ds, position[0], position[1])
    meta = SampleMeta(
        sensor_id=ds.attrs.get("sensor", "enmap"),
        row=int(position[0]),
        col=int(position[1]),
        datetime=ds.attrs.get("datetime"),
    )
    return Sample(spectrum=spectrum, meta=meta)
