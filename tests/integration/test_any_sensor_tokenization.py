from __future__ import annotations

import numpy as np

from alchemi.data.cube import Cube
from alchemi.srf.utils import load_sensor_srf

SENSORS = ("emit", "enmap", "avirisng", "hytes")


def _cube_for_sensor(sensor_id: str) -> Cube:
    srf = load_sensor_srf(sensor_id)
    assert srf is not None
    axis = srf.centers_nm.copy()
    data = np.random.default_rng(hash(sensor_id) & 0xFFFF).standard_normal((4, 3, axis.shape[0]))
    value_kind = "brightness_temp" if sensor_id == "hytes" else "radiance"
    return Cube(
        data=data,
        axis=axis,
        axis_unit="wavelength_nm",
        value_kind=value_kind,
        srf_id=sensor_id,
    )


def test_tokenization_pipeline_supports_known_sensors():
    for sensor in SENSORS:
        cube = _cube_for_sensor(sensor)
        tokens = cube.to_tokens()
        assert tokens.bands.shape[0] == cube.band_count
        assert tokens.meta.axis_unit == "nm"
        assert np.all(np.diff(cube.axis) > 0)
        assert np.all(np.isfinite(tokens.bands))
