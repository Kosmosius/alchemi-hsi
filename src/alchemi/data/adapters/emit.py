"""Adapter utilities for EMIT hyperspectral radiance cubes."""

from __future__ import annotations

from alchemi.data.io import emit_pixel as _emit_pixel
from alchemi.data.io import load_emit_l1b

__all__ = ["load_emit_l1b", "load_emit_pixel"]


def load_emit_pixel(path: str, y: int, x: int, **kwargs):
    """Load a single EMIT pixel as a :class:`~alchemi.types.Spectrum`."""

    dataset = load_emit_l1b(path, **kwargs)
    return _emit_pixel(dataset, y, x)
