"""Public ingestion entrypoints and lightweight adapters that convert IO datasets
into :class:`~alchemi.data.cube.Cube`."""

from __future__ import annotations

from .avirisng import from_avirisng_l1b
from .emit import from_emit_l1b
from .enmap import from_enmap_l1b
from .hytes import from_hytes_bt
from .mako import from_mako_l2s, from_mako_l3
from alchemi.io.mako import (
    ACE_GAS_NAMES,
    MAKO_BAND_COUNT,
    mako_pixel_bt,
    mako_pixel_radiance,
    open_mako_ace,
    open_mako_btemp,
    open_mako_l2s,
)

__all__ = [
    "from_avirisng_l1b",
    "from_emit_l1b",
    "from_enmap_l1b",
    "from_hytes_bt",
    "from_mako_l2s",
    "from_mako_l3",
    "ACE_GAS_NAMES",
    "MAKO_BAND_COUNT",
    "mako_pixel_bt",
    "mako_pixel_radiance",
    "open_mako_ace",
    "open_mako_btemp",
    "open_mako_l2s",
]
