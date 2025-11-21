from __future__ import annotations

import math

import numpy as np

from alchemi.data.cube import Cube


def test_iter_tiles_covers_image_without_overlap() -> None:
    height, width, bands = 10, 10, 3
    data = np.arange(height * width * bands, dtype=np.float32).reshape(height, width, bands)
    cube = Cube(data=data, axis=np.linspace(400.0, 800.0, bands), axis_unit="wavelength_nm", value_kind="radiance")

    coverage = np.zeros((height, width), dtype=np.int32)
    tile_shapes: set[tuple[int, int]] = set()

    tiles = list(cube.iter_tiles(tile_h=4, tile_w=4))

    expected_tiles = math.ceil(height / 4) * math.ceil(width / 4)
    assert len(tiles) == expected_tiles

    expected_shapes = {
        (min(4, height - row), min(4, width - col))
        for row in range(0, height, 4)
        for col in range(0, width, 4)
    }

    for row_slice, col_slice, subcube in tiles:
        coverage[row_slice, col_slice] += 1
        tile_shapes.add(subcube.data.shape[:2])

        assert subcube.data.shape[2] == cube.band_count
        assert np.shares_memory(subcube.data, cube.data)
        np.testing.assert_allclose(subcube.axis, cube.axis)

        assert subcube.data.shape[0] == row_slice.stop - row_slice.start
        assert subcube.data.shape[1] == col_slice.stop - col_slice.start

    np.testing.assert_array_equal(coverage, np.ones_like(coverage))
    assert tile_shapes == expected_shapes
