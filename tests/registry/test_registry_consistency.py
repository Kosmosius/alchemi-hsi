import json
from pathlib import Path

import numpy as np

from alchemi.registry import sensors, srfs


def test_registered_sensors_have_consistent_specs():
    registry = sensors._seed_registry()
    for sensor_id in registry.list_sensors():
        spec = registry.get_sensor(sensor_id)
        assert spec.band_centers_nm.shape[0] == spec.expected_band_count
        assert spec.band_widths_nm.shape == spec.band_centers_nm.shape
        assert spec.wavelength_range_nm[0] < spec.wavelength_range_nm[1]


def test_get_srf_normalizes_rows(monkeypatch, tmp_path):
    monkeypatch.setattr(srfs, "_SRF_ROOT", Path(tmp_path))
    centers = np.array([500.0, 600.0])
    bands = [
        {"nm": [490.0, 500.0, 510.0], "resp": [0.2, 0.6, 0.2]},
        {"nm": [590.0, 600.0, 610.0], "resp": [1.0, 1.0, 1.0]},
    ]
    payload = {"sensor": "toy", "centers_nm": centers.tolist(), "bands": bands}
    path = Path(tmp_path) / "toy_srfs.json"
    path.write_text(json.dumps(payload))

    srf = srfs.get_srf("toy")
    assert srf.centers_nm.shape[0] == 2
    for nm, resp in zip(srf.bands_nm, srf.bands_resp, strict=True):
        integral = np.trapz(resp, x=nm)
        assert np.isclose(integral, 1.0)
