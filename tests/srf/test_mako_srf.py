import numpy as np

from alchemi.srf import build_mako_srf_from_header, get_srf, mako_lwir_grid_nm
from alchemi.srf.resample import project_to_sensor


def _trapz(values: np.ndarray, grid: np.ndarray) -> float:
    try:
        return float(np.trapezoid(values, grid))
    except AttributeError:  # pragma: no cover - NumPy < 2.0 fallback
        return float(np.trapz(values, grid))


def _comex_header_wavelengths() -> np.ndarray:
    return np.linspace(7600.0, 13200.0, 128, dtype=np.float64)


def test_mako_srf_integral_1() -> None:
    wavelengths = _comex_header_wavelengths()
    grid = mako_lwir_grid_nm()
    srf = build_mako_srf_from_header(wavelengths)

    for nm, resp in zip(srf.bands_nm, srf.bands_resp, strict=True):
        np.testing.assert_allclose(nm, grid)
        area = _trapz(resp, grid)
        assert np.isclose(area, 1.0, atol=1e-6)


def test_mako_srf_flat_response() -> None:
    wavelengths = _comex_header_wavelengths()
    grid = mako_lwir_grid_nm()
    srf = build_mako_srf_from_header(wavelengths)

    flat = np.ones_like(grid, dtype=np.float64)
    projected = project_to_sensor(grid, flat, srf.centers_nm, srf=srf)
    assert projected.shape == (wavelengths.size,)
    assert np.allclose(projected, 1.0, atol=1e-6)


def test_mako_srf_centers_match_header() -> None:
    wavelengths = _comex_header_wavelengths()
    grid = mako_lwir_grid_nm()
    srf = build_mako_srf_from_header(wavelengths)

    effective = [_trapz(grid * resp, grid) for resp in srf.bands_resp]
    effective_nm = np.asarray(effective)
    assert np.allclose(effective_nm, wavelengths, atol=5.0)


def test_mako_registry_default_grid() -> None:
    srf, grid = get_srf("mako")
    assert srf.sensor == "mako"
    np.testing.assert_allclose(grid, mako_lwir_grid_nm())
    np.testing.assert_allclose(srf.centers_nm, _comex_header_wavelengths())
