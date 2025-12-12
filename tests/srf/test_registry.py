import numpy as np

from alchemi.registry import srfs
from alchemi.spectral.srf import SensorSRF, SRFProvenance
from alchemi.srf import avirisng, emit, enmap, hytes, mako  # noqa: F401
from alchemi.srf.registry import GLOBAL_SRF_REGISTRY, SRFRegistry


def test_registry_roundtrip():
    grid = np.linspace(400, 410, 5, dtype=np.float64)
    srfs = np.stack([np.linspace(0, 1, 5), np.linspace(1, 0, 5)], axis=0)
    srfs = srfs / np.trapz(srfs, x=grid, axis=1)[:, None]
    payload = SensorSRF(
        sensor_id="dummy",
        wavelength_grid_nm=grid,
        srfs=srfs,
        band_centers_nm=np.array([402.5, 407.5], dtype=np.float64),
        band_widths_nm=np.array([5.0, 5.0], dtype=np.float64),
        provenance=SRFProvenance.GAUSSIAN,
    )

    reg = SRFRegistry()
    reg.register(payload)

    assert reg.has("dummy")
    registered = reg.get("DUMMY")
    assert registered is not None
    assert registered.sensor.lower() == "dummy"
    np.testing.assert_allclose(registered.centers_nm, payload.band_centers_nm)


def test_global_registry_populated():
    emit_srf = GLOBAL_SRF_REGISTRY.get("emit")
    enmap_srf = GLOBAL_SRF_REGISTRY.get("enmap")
    hytes_srf = GLOBAL_SRF_REGISTRY.get("hytes")

    for sensor in (emit_srf, enmap_srf, hytes_srf):
        assert sensor is not None
        assert len(sensor.bands_resp) == len(sensor.centers_nm)
        assert np.asarray(sensor.bands_resp).ndim == 2
        assert np.asarray(sensor.bands_nm).ndim == 2


def test_public_registry_returns_canonical_srfs():
    sensors = ("emit", "enmap", "avirisng", "hytes")
    loaded = []

    for sensor in sensors:
        canonical = srfs.get_sensor_srf(sensor)
        assert isinstance(canonical, SensorSRF)
        assert canonical.band_centers_nm.size > 0
        loaded.append(canonical)

    ids = {s.sensor_id.lower() for s in loaded}
    assert len(ids) == len(sensors)
    assert srfs.get_sensor_srf("aviris-ng").sensor_id.lower() in ids
