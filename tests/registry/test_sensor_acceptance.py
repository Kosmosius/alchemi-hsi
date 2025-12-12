import numpy as np

from alchemi.registry.acceptance import (
    AcceptanceVerdict,
    _estimate_fwhm,
    evaluate_sensor_acceptance,
)
from alchemi.registry.sensors import DEFAULT_SENSOR_REGISTRY, SensorSpec
from alchemi.types import SRFMatrix


def _make_triangular_band(center: float, width: float) -> tuple[np.ndarray, np.ndarray]:
    nm = np.linspace(center - width, center + width, 5, dtype=np.float64)
    resp = np.array([0.0, 0.5, 1.0, 0.5, 0.0], dtype=np.float64)
    return nm, resp


def _build_srf(sensor_id: str, centers: np.ndarray, widths: np.ndarray) -> SRFMatrix:
    bands_nm: list[np.ndarray] = []
    bands_resp: list[np.ndarray] = []
    for center, width in zip(centers, widths, strict=True):
        nm, resp = _make_triangular_band(center, width)
        bands_nm.append(nm)
        bands_resp.append(resp)
    return SRFMatrix(sensor=sensor_id, centers_nm=centers, bands_nm=bands_nm, bands_resp=bands_resp)


def test_estimate_fwhm_matches_triangular_width():
    nm, resp = _make_triangular_band(center=500.0, width=20.0)
    assert _estimate_fwhm(nm, resp) == 20.0


def test_evaluate_sensor_acceptance_accepted_when_aligned():
    centers = np.array([410.0, 430.0, 450.0])
    widths = np.array([10.0, 10.0, 10.0])
    spec = SensorSpec(
        sensor_id="test-sensor",
        expected_band_count=3,
        wavelength_range_nm=(400.0, 500.0),
        band_centers_nm=centers,
        band_widths_nm=widths,
        srf_source="official",
        absorption_windows_nm=[(405.0, 415.0)],
    )
    srf_matrix = _build_srf(spec.sensor_id, centers, widths)

    report = evaluate_sensor_acceptance(spec, srf_matrix)
    assert report.verdict is AcceptanceVerdict.ACCEPT


def test_evaluate_sensor_acceptance_flags_center_mismatch():
    centers = np.array([410.0, 430.0, 450.0])
    widths = np.full_like(centers, 10.0)
    spec = SensorSpec(
        sensor_id="center-mismatch",
        expected_band_count=3,
        wavelength_range_nm=(400.0, 500.0),
        band_centers_nm=centers,
        band_widths_nm=widths,
        srf_source="official",
    )
    shifted_centers = centers + 20.0
    srf_matrix = _build_srf(spec.sensor_id, shifted_centers, widths)

    report = evaluate_sensor_acceptance(spec, srf_matrix)
    assert report.verdict is AcceptanceVerdict.REJECT


def test_evaluate_sensor_acceptance_flags_width_mismatch():
    centers = np.array([410.0, 430.0, 450.0])
    spec_widths = np.array([10.0, 10.0, 10.0])
    spec = SensorSpec(
        sensor_id="width-mismatch",
        expected_band_count=3,
        wavelength_range_nm=(400.0, 500.0),
        band_centers_nm=centers,
        band_widths_nm=spec_widths,
        srf_source="official",
    )
    wide_widths = np.array([40.0, 40.0, 40.0])
    srf_matrix = _build_srf(spec.sensor_id, centers, wide_widths)

    report = evaluate_sensor_acceptance(spec, srf_matrix)
    assert report.verdict is AcceptanceVerdict.REJECT


def test_evaluate_sensor_acceptance_flags_absorption_window_gap():
    centers = np.array([410.0, 420.0, 430.0])
    widths = np.full_like(centers, 10.0)
    spec = SensorSpec(
        sensor_id="window-gap",
        expected_band_count=3,
        wavelength_range_nm=(400.0, 460.0),
        band_centers_nm=centers,
        band_widths_nm=widths,
        srf_source="official",
        absorption_windows_nm=[(440.0, 445.0)],
    )
    srf_matrix = _build_srf(spec.sensor_id, centers, widths)

    report = evaluate_sensor_acceptance(spec, srf_matrix)
    assert report.verdict is AcceptanceVerdict.ACCEPT_WITH_WARNINGS


def test_evaluate_sensor_acceptance_flags_out_of_scope_coverage():
    centers = np.array([410.0, 430.0, 450.0])
    widths = np.full_like(centers, 10.0)
    spec = SensorSpec(
        sensor_id="coverage-gap",
        expected_band_count=3,
        wavelength_range_nm=(400.0, 500.0),
        band_centers_nm=centers,
        band_widths_nm=widths,
        srf_source="official",
    )
    narrow_centers = np.array([470.0, 480.0, 490.0])
    srf_matrix = _build_srf(spec.sensor_id, narrow_centers, widths)

    report = evaluate_sensor_acceptance(spec, srf_matrix)
    assert report.verdict is AcceptanceVerdict.REJECT


def test_known_sensor_configs_are_accepted():
    for sensor_id in ["emit", "enmap", "aviris-ng", "hytes"]:
        spec = DEFAULT_SENSOR_REGISTRY.get_sensor(sensor_id)
        srf = _build_srf(spec.sensor_id, spec.band_centers_nm, spec.band_widths_nm)
        report = evaluate_sensor_acceptance(spec, srf)
        assert report.verdict is AcceptanceVerdict.ACCEPT


def test_multi_lobed_srf_is_rejected():
    centers = np.array([1000.0, 1100.0, 1200.0])
    widths = np.full_like(centers, 10.0)
    spec = SensorSpec(
        sensor_id="lobed",
        expected_band_count=3,
        wavelength_range_nm=(900.0, 1300.0),
        band_centers_nm=centers,
        band_widths_nm=widths,
        srf_source="synthetic",
    )

    bands_nm: list[np.ndarray] = []
    bands_resp: list[np.ndarray] = []
    for center in centers:
        nm = np.linspace(center - 10.0, center + 10.0, 7, dtype=np.float64)
        resp = np.array([0.0, 0.6, 0.1, 0.7, 0.1, 0.6, 0.0], dtype=np.float64)
        bands_nm.append(nm)
        bands_resp.append(resp)
    srf_matrix = SRFMatrix(sensor="lobed", centers_nm=centers, bands_nm=bands_nm, bands_resp=bands_resp)

    report = evaluate_sensor_acceptance(spec, srf_matrix)
    assert report.verdict is AcceptanceVerdict.REJECT


def test_extreme_gaps_trigger_rejection():
    centers = np.array([410.0, 800.0, 2400.0])
    widths = np.array([10.0, 10.0, 10.0])
    spec = SensorSpec(
        sensor_id="gappy",
        expected_band_count=3,
        wavelength_range_nm=(400.0, 2500.0),
        band_centers_nm=centers,
        band_widths_nm=widths,
        srf_source="synthetic",
    )
    srf_matrix = _build_srf(spec.sensor_id, centers, widths)

    report = evaluate_sensor_acceptance(spec, srf_matrix)
    assert report.verdict is AcceptanceVerdict.REJECT


def test_overwide_bands_rejected():
    centers = np.array([1000.0, 1100.0, 1200.0])
    widths = np.array([200.0, 200.0, 200.0])
    spec = SensorSpec(
        sensor_id="overwide",
        expected_band_count=3,
        wavelength_range_nm=(900.0, 1300.0),
        band_centers_nm=centers,
        band_widths_nm=np.full_like(centers, 10.0),
        srf_source="synthetic",
    )
    srf_matrix = _build_srf(spec.sensor_id, centers, widths)

    report = evaluate_sensor_acceptance(spec, srf_matrix)
    assert report.verdict is AcceptanceVerdict.REJECT


def test_narrow_bands_warn_not_reject():
    centers = np.array([500.0, 510.0, 520.0, 530.0, 540.0])
    widths = np.full_like(centers, 2.5)
    spec_widths = np.full_like(centers, 4.0)
    spec = SensorSpec(
        sensor_id="narrow",
        expected_band_count=centers.size,
        wavelength_range_nm=(480.0, 560.0),
        band_centers_nm=centers,
        band_widths_nm=spec_widths,
        srf_source="synthetic",
    )
    srf_matrix = _build_srf(spec.sensor_id, centers, widths)

    report = evaluate_sensor_acceptance(spec, srf_matrix)
    assert report.verdict is AcceptanceVerdict.ACCEPT_WITH_WARNINGS
