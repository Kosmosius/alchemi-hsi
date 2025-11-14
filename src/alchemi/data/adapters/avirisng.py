"""Adapter utilities for AVIRIS-NG samples."""

from __future__ import annotations

import xarray as xr

from alchemi.types import Sample, SampleMeta
from alchemi.data.io import avirisng_pixel

__all__ = ["load_avirisng_pixel"]


def load_avirisng_pixel(ds: xr.Dataset, position: tuple[int, int]) -> Sample:
    """Create a :class:`~alchemi.types.Sample` from a dataset and (y, x) index."""

    y, x = position
    spectrum = avirisng_pixel(ds, y=y, x=x)
    meta = SampleMeta(
        sensor_id=ds.attrs.get("sensor", "avirisng"),
        row=int(y),
        col=int(x),
        datetime=ds.attrs.get("datetime"),
    )
    return Sample(spectrum=spectrum, meta=meta)

