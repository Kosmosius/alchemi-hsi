import numpy as np

from alchemi.srf.utils import resolve_band_widths
from alchemi.spectral.srf import SensorSRF, SRFProvenance


def _toy_sensor_srf(centers: np.ndarray, widths: np.ndarray) -> SensorSRF:
    grid = centers
    srfs = []
    for center, width in zip(centers, widths, strict=True):
        half = width / 2.0
        resp = np.zeros_like(grid)
        resp[(grid >= center - half) & (grid <= center + half)] = 1.0
        area = np.trapz(resp, x=grid)
        resp = resp / area if area > 0 else resp
        srfs.append(resp)
    return SensorSRF(
        sensor_id="toy",
        wavelength_grid_nm=grid,
        srfs=np.asarray(srfs),
        band_centers_nm=centers,
        band_widths_nm=widths,
        provenance=SRFProvenance.OFFICIAL,
    )


def test_resolve_band_widths_prefers_srf_band_widths():
    centers = np.linspace(400.0, 700.0, 5)
    widths = np.linspace(5.0, 9.0, centers.size)
    srf = _toy_sensor_srf(centers, widths)

    resolved, default_mask, source = resolve_band_widths("toy", centers, srf=srf)

    assert source == "srf"
    assert not np.any(default_mask)
    assert np.allclose(resolved, widths)


def test_resolve_band_widths_uses_fwhm_when_missing_srf():
    axis = np.array([500.0, 510.0, 520.0])
    fwhm = np.array([6.0, 7.0, 8.0])

    resolved, default_mask, source = resolve_band_widths("unknown", axis, fwhm=fwhm)

    assert source == "fwhm"
    assert not np.any(default_mask)
    assert np.allclose(resolved, fwhm)


def test_resolve_band_widths_marks_default_when_no_metadata():
    axis = np.array([1.0, 3.0, 6.0])

    resolved, default_mask, source = resolve_band_widths(None, axis)

    assert source == "default"
    assert np.all(default_mask)
    assert resolved.shape == axis.shape
    assert np.all(resolved > 0)
