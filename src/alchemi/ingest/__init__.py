"""Lightweight adapters that convert IO datasets into :class:`~alchemi.data.cube.Cube`."""

from __future__ import annotations

from .avirisng import from_avirisng_l1b
from .emit import from_emit_l1b
from .enmap import from_enmap_l1b
from .hytes import from_hytes_bt
from .mako import from_mako_l2s, from_mako_l3

__all__ = [
    "from_avirisng_l1b",
    "from_emit_l1b",
    "from_enmap_l1b",
    "from_hytes_bt",
    "from_mako_l2s",
    "from_mako_l3",
]
