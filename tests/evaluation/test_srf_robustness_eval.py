import numpy as np

from alchemi.evaluation.srf_robustness_eval import sweep_perturbations
from alchemi.srf.synthetic import make_virtual_sensor


def test_axis_mode_runs_on_toy_data() -> None:
    spectra = np.stack(
        [np.linspace(0, 1, 8, dtype=np.float64), np.linspace(1, 0, 8, dtype=np.float64)]
    )

    settings = [{"center_shift": 0.05, "fwhm_scale": 1.1}]
    results = sweep_perturbations(spectra, settings, perturbation_mode="axis")

    assert len(results) == 1
    assert "degradation" in results[0]
    assert results[0]["degradation"] >= 0.0


def test_srf_mode_perturbs_centers_and_widths() -> None:
    wavelength_grid_nm = np.linspace(400.0, 420.0, 81, dtype=np.float64)
    sensor = make_virtual_sensor(
        wavelength_min_nm=float(wavelength_grid_nm[0]),
        wavelength_max_nm=float(wavelength_grid_nm[-1]),
        band_count=5,
        grid_step_nm=float(np.diff(wavelength_grid_nm).mean()),
        rng=np.random.default_rng(0),
    )

    highres_spectra = np.stack(
        [np.ones_like(wavelength_grid_nm), np.linspace(0.2, 1.2, wavelength_grid_nm.size)]
    )

    settings = [
        {
            "center_shift_std_nm": 2.0,
            "width_scale_std": 0.2,
            "shape_jitter_std": 0.05,
            "seed": 123,
        }
    ]

    results = sweep_perturbations(
        highres_spectra,
        settings,
        perturbation_mode="srf",
        wavelength_grid_nm=wavelength_grid_nm,
        base_sensor_srf=sensor,
    )

    assert "perturbed_centers_nm" in results[0]
    assert "perturbed_widths_nm" in results[0]

    centers_delta = np.abs(results[0]["perturbed_centers_nm"] - sensor.band_centers_nm)
    widths_delta = np.abs(results[0]["perturbed_widths_nm"] - sensor.band_widths_nm)

    assert np.any(centers_delta > 0.0)
    assert np.any(widths_delta > 0.0)
    assert results[0]["degradation"] >= 0.0
