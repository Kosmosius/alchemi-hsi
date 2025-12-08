import logging

import numpy as np
import pytest

from alchemi.data.cube import Cube
from alchemi.data.validators import check_cube_health
from alchemi.types import QuantityKind


def _base_cube(kind: QuantityKind = QuantityKind.RADIANCE) -> Cube:
    data = np.ones((2, 2, 3), dtype=float)
    axis = np.array([400.0, 500.0, 600.0], dtype=float)
    return Cube(data=data, axis=axis, axis_unit="wavelength_nm", value_kind=kind)


def test_valid_cube_passes_health_check():
    cube = _base_cube()
    check_cube_health(cube)


def test_non_monotonic_axis_raises():
    cube = _base_cube()
    cube.axis = cube.axis[::-1]

    with pytest.raises(ValueError):
        check_cube_health(cube)


def test_band_mask_mismatch_raises():
    cube = _base_cube()
    cube.band_mask = np.zeros(4, dtype=bool)

    with pytest.raises(ValueError):
        check_cube_health(cube)


def test_radiance_outlier_logs_warning(caplog):
    cube = _base_cube()
    cube.data[0, 0, 0] = 2e5

    caplog.set_level(logging.WARNING)
    check_cube_health(cube, logger=logging.getLogger("alchemi.test"))

    assert any("Radiance cube has very large values" in rec.message for rec in caplog.records)


def test_reflectance_outlier_logs_warning(caplog):
    cube = _base_cube(QuantityKind.REFLECTANCE)
    cube.data[..., 1] = 2.0

    caplog.set_level(logging.WARNING)
    check_cube_health(cube, logger=logging.getLogger("alchemi.test"))

    assert any("Reflectance cube 99th percentile outside" in rec.message for rec in caplog.records)

