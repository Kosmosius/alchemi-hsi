"""Adapter utilities for COMEX Mako samples."""

from __future__ import annotations

import xarray as xr

from alchemi.io import mako_pixel_bt, mako_pixel_radiance
from alchemi.types import Sample, SampleMeta

__all__ = ["load_mako_pixel", "load_mako_pixel_bt"]


def load_mako_pixel(ds: xr.Dataset, position: tuple[int, int]) -> Sample:
    """Create a :class:`~alchemi.types.Sample` from a Mako L2S dataset."""

    spectrum = mako_pixel_radiance(ds, position[0], position[1])
    meta = _build_meta(ds, position)
    return Sample(spectrum=spectrum, meta=meta)


def load_mako_pixel_bt(ds: xr.Dataset, position: tuple[int, int]) -> Sample:
    """Create a :class:`~alchemi.types.Sample` from a Mako BTEMP dataset."""

    spectrum = mako_pixel_bt(ds, position[0], position[1])
    meta = _build_meta(ds, position)
    return Sample(spectrum=spectrum, meta=meta)


def _build_meta(ds: xr.Dataset, position: tuple[int, int]) -> SampleMeta:
    return SampleMeta(
        sensor_id="mako",
        row=int(position[0]),
        col=int(position[1]),
        datetime=ds.attrs.get("datetime"),
    )
