import numpy as np
import pytest

from alchemi.data.cube import Cube
from alchemi.data.sample_utils import cube_from_sample
from alchemi.types import SpectrumKind


def test_sample_from_cube_roundtrip():
    axis = np.array([500.0, 600.0, 700.0], dtype=np.float64)
    data = np.array(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ],
        dtype=np.float64,
    )

    cube = Cube(
        data=data,
        axis=axis,
        axis_unit="wavelength_nm",
        value_kind="radiance",
        attrs={"sensor": "test-sensor", "units": "W"},
    )

    sample = cube.sample_at(1, 0)

    np.testing.assert_array_equal(sample.spectrum.values, data[1, 0, :])
    np.testing.assert_array_equal(sample.spectrum.wavelengths.nm, axis)
    assert sample.spectrum.kind is SpectrumKind.RADIANCE
    assert sample.spectrum.units == "W"
    assert sample.meta["sensor"] == "test-sensor"
    assert sample.meta["row"] == 1
    assert sample.meta["col"] == 0

    rebuilt_cube = cube_from_sample(sample)
    assert rebuilt_cube.shape == (1, 1, 3)
    np.testing.assert_array_equal(rebuilt_cube.data[0, 0, :], data[1, 0, :])
    np.testing.assert_array_equal(rebuilt_cube.axis, axis)
    assert rebuilt_cube.value_kind == "radiance"
    assert rebuilt_cube.sensor == "test-sensor"
    assert rebuilt_cube.units == "W"


def test_sample_at_out_of_bounds():
    cube = Cube(
        data=np.ones((2, 2, 2)),
        axis=np.array([1.0, 2.0]),
        axis_unit="wavelength_nm",
        value_kind="radiance",
    )

    with pytest.raises(IndexError):
        cube.sample_at(2, 0)
