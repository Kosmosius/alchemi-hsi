"""Adapter utilities for COMEX Mako samples."""

from __future__ import annotations

import xarray as xr

from alchemi.types import Sample, SampleMeta
from alchemi.io import mako_pixel_radiance

__all__ = ["load_mako_pixel"]


def load_mako_pixel(ds: xr.Dataset, position: tuple[int, int]) -> Sample:
    """Create a :class:`~alchemi.types.Sample` from a Mako L2S dataset."""

    spectrum = mako_pixel_radiance(ds, position[0], position[1])
    meta = SampleMeta(
        sensor_id="mako",
        row=int(position[0]),
        col=int(position[1]),
        datetime=ds.attrs.get("datetime"),
    )
    return Sample(spectrum=spectrum, meta=meta)

