import numpy as np

from alchemi.physics.planck import (
    band_averaged_radiance,
    bt_spectrum_to_radiance,
    invert_band_averaged_radiance_to_bt,
    inverse_planck_central_lambda,
    planck_radiance_wavelength,
    radiance_spectrum_to_bt,
)
from alchemi.types import Spectrum, WavelengthGrid


def _gaussian_srfs(
    wavelength_nm: np.ndarray, centers_nm: list[float], sigma_nm: float
) -> np.ndarray:
    srfs = []
    for center in centers_nm:
        resp = np.exp(-0.5 * ((wavelength_nm - center) / sigma_nm) ** 2)
        area = float(np.trapezoid(resp, x=wavelength_nm))
        srfs.append(resp / area)
    return np.asarray(srfs)


def test_planck_radiance_matches_reference():
    wl_nm = 10_000.0  # 10 µm
    radiance = planck_radiance_wavelength(wl_nm, 300.0)
    assert np.isclose(radiance, 9.924e-3, rtol=1e-4, atol=5e-6)


def test_planck_radiance_monotonic_in_temperature():
    wl_nm = 10_000.0
    temps = np.array([260.0, 280.0, 300.0])
    radiance = planck_radiance_wavelength(wl_nm, temps)
    assert np.all(np.diff(radiance) > 0)


def test_inverse_planck_roundtrip():
    wl_nm = np.array([8_000.0, 10_000.0, 12_000.0])
    temps = np.linspace(200.0, 350.0, 6)
    radiance = planck_radiance_wavelength(wl_nm[None, :], temps[:, None])
    recovered = inverse_planck_central_lambda(radiance, wl_nm[None, :])
    assert np.allclose(recovered, temps[:, None], atol=1e-2)


def test_band_averaged_inversion_matches_true_temperature():
    wl_grid = np.linspace(8_000.0, 12_000.0, 400)
    srfs = _gaussian_srfs(wl_grid, [8_700.0, 10_000.0, 11_300.0], sigma_nm=80.0)
    true_temp = 310.0
    band_radiance = band_averaged_radiance(true_temp, srfs, wl_grid)
    recovered = invert_band_averaged_radiance_to_bt(band_radiance, srfs, wl_grid)
    assert np.allclose(recovered, true_temp, atol=1e-2)


def test_spectrum_central_lambda_roundtrip():
    wl_nm = np.linspace(8_000.0, 12_000.0, 5)
    temps = np.linspace(285.0, 315.0, wl_nm.size)
    radiance = planck_radiance_wavelength(wl_nm, temps)
    spectrum = Spectrum.from_radiance(WavelengthGrid(wl_nm), radiance)

    bt_spec = radiance_spectrum_to_bt(spectrum, method="central_lambda")
    assert np.allclose(bt_spec.values, temps, atol=1e-2)

    recovered = bt_spectrum_to_radiance(bt_spec, method="central_lambda")
    assert np.allclose(recovered.values, radiance, rtol=1e-4, atol=1e-6)


def test_spectrum_band_method_roundtrip():
    wl_grid = np.linspace(8_000.0, 12_000.0, 500)
    srfs = _gaussian_srfs(wl_grid, [8_500.0, 9_800.0, 11_200.0], sigma_nm=60.0)
    lambda_eff = np.trapezoid(srfs * wl_grid[None, :], x=wl_grid, axis=1) / np.trapezoid(
        srfs, x=wl_grid, axis=1
    )

    temps = np.array([295.0, 305.0, 315.0])
    band_radiance = band_averaged_radiance(temps, srfs, wl_grid)
    spectrum = Spectrum.from_radiance(WavelengthGrid(lambda_eff), band_radiance)

    bt_spec = radiance_spectrum_to_bt(
        spectrum,
        srf_matrix=srfs,
        srf_wavelength_nm=wl_grid,
        method="band",
    )
    assert np.allclose(bt_spec.values, temps, atol=1e-2)

    recovered = bt_spectrum_to_radiance(
        bt_spec,
        srf_matrix=srfs,
        srf_wavelength_nm=wl_grid,
        method="band",
    )
    assert np.allclose(recovered.values, band_radiance, rtol=1e-4, atol=1e-6)
