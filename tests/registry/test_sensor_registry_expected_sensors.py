import numpy as np

from alchemi.registry.sensors import DEFAULT_SENSOR_REGISTRY


EXPECTED_SENSORS = ["emit", "enmap", "aviris-ng", "hytes"]


def test_default_registry_contains_expected_sensors():
    assert DEFAULT_SENSOR_REGISTRY.list_sensors() == sorted(EXPECTED_SENSORS)


def test_default_sensor_specs_are_well_formed():
    for sensor_id in EXPECTED_SENSORS:
        spec = DEFAULT_SENSOR_REGISTRY.get_sensor(sensor_id)

        assert spec.expected_band_count == spec.band_centers_nm.shape[0]

        centers = spec.band_centers_nm
        assert np.all(np.diff(centers) > 0)
        assert centers.min() >= spec.wavelength_range_nm[0]
        assert centers.max() <= spec.wavelength_range_nm[1]

        widths = spec.band_widths_nm
        assert widths.ndim == 1
        assert widths.shape == centers.shape

        if spec.bad_band_mask is not None:
            assert spec.bad_band_mask.shape == centers.shape

        if spec.absorption_windows_nm:
            for start, end in spec.absorption_windows_nm:
                assert spec.wavelength_range_nm[0] <= start < end <= spec.wavelength_range_nm[1]
