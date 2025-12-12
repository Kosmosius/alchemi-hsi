import numpy as np

from alchemi.srf.synthetic import SRFJitterConfig, jitter_sensor_srf, make_gaussian_srf
from alchemi.spectral.srf import SensorSRF


def _dummy_sensor() -> SensorSRF:
    grid = np.linspace(400.0, 500.0, 25, dtype=np.float64)
    centers = np.array([420.0, 470.0], dtype=np.float64)
    widths = np.array([10.0, 12.0], dtype=np.float64)
    srfs = make_gaussian_srf(centers, widths, wavelength_grid_nm=grid)
    valid_mask = np.array([True, False])
    return SensorSRF(
        sensor_id="dummy",
        wavelength_grid_nm=grid,
        srfs=srfs,
        band_centers_nm=centers,
        band_widths_nm=widths,
        valid_mask=valid_mask,
    )


def test_jitter_disabled_is_identity() -> None:
    sensor = _dummy_sensor()
    cfg = SRFJitterConfig(enabled=False, seed=123)
    jittered = jitter_sensor_srf(sensor, cfg)

    assert jittered is sensor


def test_jitter_preserves_invariants() -> None:
    sensor = _dummy_sensor()
    cfg = SRFJitterConfig(
        enabled=True,
        center_shift_std_nm=2.5,
        width_scale_std=0.15,
        shape_jitter_std=0.05,
        seed=7,
    )

    jittered = jitter_sensor_srf(sensor, cfg)

    assert jittered is not sensor
    centers_delta = np.abs(jittered.band_centers_nm - sensor.band_centers_nm)
    assert centers_delta.max() > 0
    assert centers_delta.max() < cfg.center_shift_std_nm * 5 + 1e-6
    assert np.all(np.diff(jittered.band_centers_nm) > 0)

    width_ratio = jittered.band_widths_nm / sensor.band_widths_nm
    assert np.all(width_ratio > 0)
    assert width_ratio.max() < np.exp(cfg.width_scale_std * 5) + 1e-6

    assert np.array_equal(jittered.valid_mask, sensor.valid_mask)
    assert jittered.meta.get("jitter") is not None

    row_area = np.trapezoid(jittered.srfs, x=jittered.wavelength_grid_nm, axis=1)
    np.testing.assert_allclose(row_area, 1.0, atol=1e-6, rtol=1e-6)
    assert np.all(jittered.srfs >= 0)

    assert not np.allclose(sensor.srfs, jittered.srfs)
