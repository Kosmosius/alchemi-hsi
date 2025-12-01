"""Utilities for tiling large scenes into manageable chips."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class Chip:
    """Represents a spatial chip cut from a larger cube."""

    data: NDArray[np.floating]
    top_left: tuple[int, int]

    def to_global(self, row: int, col: int) -> tuple[int, int]:
        """Convert chip-relative coordinates to scene coordinates."""

        return self.top_left[0] + row, self.top_left[1] + col


def iter_tiles(
    cube: NDArray[np.floating],
    patch_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
) -> Iterator[Chip]:
    """Yield :class:`Chip` objects from a cube.

    Parameters
    ----------
    cube:
        3-D array shaped ``(H, W, Bands)``.
    patch_size:
        Scalar or ``(height, width)`` tuple describing chip dimensions.
    stride:
        Stride between chips. Defaults to non-overlapping chips.
    """

    if cube.ndim != 3:
        raise ValueError("cube must be 3-D (H, W, Bands)")

    h, w, _ = cube.shape
    ph, pw = _normalize_shape(patch_size)
    sh, sw = _normalize_shape(stride or patch_size)

    for y in range(0, max(h - ph + 1, 1), sh):
        for x in range(0, max(w - pw + 1, 1), sw):
            window = cube[y : y + ph, x : x + pw, :]
            yield Chip(data=window.copy(), top_left=(y, x))


def compute_tile_origins(
    scene_shape: Sequence[int], patch_size: int | tuple[int, int], stride: int | tuple[int, int]
) -> list[tuple[int, int]]:
    """Return top-left coordinates for all tiles in a scene."""

    h, w = int(scene_shape[0]), int(scene_shape[1])
    ph, pw = _normalize_shape(patch_size)
    sh, sw = _normalize_shape(stride)
    origins: list[tuple[int, int]] = []
    for y in range(0, max(h - ph + 1, 1), sh):
        for x in range(0, max(w - pw + 1, 1), sw):
            origins.append((y, x))
    return origins


def _normalize_shape(size: int | tuple[int, int]) -> Tuple[int, int]:
    if isinstance(size, tuple):
        return max(1, int(size[0])), max(1, int(size[1]))
    return max(1, int(size)), max(1, int(size))
