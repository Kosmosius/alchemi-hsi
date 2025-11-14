"""Adapter utilities for HyTES samples."""

from __future__ import annotations

import xarray as xr
from alchemi.types import Sample, SampleMeta
from alchemi.data.io import hytes_pixel_bt

__all__ = ["load_hytes_pixel"]


def load_hytes_pixel(ds: xr.Dataset, position: tuple[int, int]) -> Sample:
    """Create a :class:`~alchemi.types.Sample` from a dataset and (y, x) index."""

    spectrum = hytes_pixel_bt(ds, position[0], position[1])
    meta = SampleMeta(
        sensor_id=ds.attrs.get("sensor", "hytes"),
        row=int(position[0]),
        col=int(position[1]),
        datetime=ds.attrs.get("datetime"),
    )
    return Sample(spectrum=spectrum, meta=meta)
