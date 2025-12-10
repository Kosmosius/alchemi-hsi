from pathlib import Path

import numpy as np
import pytest

from alchemi.registry import sensors, srfs


pytestmark = pytest.mark.physics_and_metadata


_DEF_TOL = dict(rtol=1e-3, atol=1e-3)


def _registered_sensor_ids() -> list[str]:
    srf_root = Path("resources/srfs")
    sensors = {
        path.name.split("_srfs", maxsplit=1)[0]
        for path in srf_root.iterdir()
        if path.is_file() and path.name.endswith(("_srfs.json", "_srfs.npy", "_srfs.npz"))
    }
    return sorted(sensors)


@pytest.mark.parametrize("sensor_id", _registered_sensor_ids())
def test_srf_rows_match_resources(sensor_id: str):
    srf = srfs.get_srf(sensor_id)
    bad_band_mask = (
        srf.bad_band_mask
        if srf.bad_band_mask is not None
        else np.zeros(len(srf.bands_resp), dtype=bool)
    )

    for nm, resp, is_bad in zip(srf.bands_nm, srf.bands_resp, bad_band_mask, strict=True):
        assert nm.size > 0
        assert np.all(np.diff(nm) > 0)
        assert nm.shape == resp.shape

        if is_bad:
            continue
        area = np.trapz(resp, x=nm)
        np.testing.assert_allclose(area, 1.0, **_DEF_TOL)


@pytest.mark.parametrize("sensor_id", _registered_sensor_ids())
def test_srf_centers_align_with_registry(sensor_id: str):
    srf = srfs.get_srf(sensor_id)
    try:
        spec = sensors.DEFAULT_SENSOR_REGISTRY.get_sensor(sensor_id)
    except KeyError:
        pytest.skip(f"No sensor spec registered for {sensor_id}")

    np.testing.assert_equal(srf.centers_nm.shape, spec.band_centers_nm.shape)
    np.testing.assert_allclose(srf.centers_nm, spec.band_centers_nm, atol=15.0)
