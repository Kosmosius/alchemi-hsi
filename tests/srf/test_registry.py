import numpy as np

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
    assert reg.get("DUMMY") is payload
    assert reg.require("dummy") is payload


def test_global_registry_populated():
    emit = GLOBAL_SRF_REGISTRY.get("emit")
    enmap = GLOBAL_SRF_REGISTRY.get("enmap")
    hytes = GLOBAL_SRF_REGISTRY.get("hytes")

    for sensor in (emit, enmap, hytes):
        assert sensor is not None
        assert sensor.srfs.ndim == 2
        assert sensor.wavelength_grid_nm.ndim == 1
        assert sensor.srfs.shape[0] == sensor.band_centers_nm.shape[0]
