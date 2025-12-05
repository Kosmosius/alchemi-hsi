import numpy as np
import pytest

from alchemi.srf.resample import interpolate_values, resample_by_interpolation
from alchemi.types import QuantityKind, Spectrum, ValueUnits


def test_interpolate_linear_monotonic_function():
    wl = np.array([400.0, 500.0, 700.0])
    values = wl**2
    targets = np.array([450.0, 600.0])

    result = interpolate_values(values, wl, targets, mode="linear")
    expected = np.interp(targets, wl, values)

    assert result.shape == (targets.shape[0],)
    np.testing.assert_allclose(result, expected)


def test_interpolate_nearest_matches_expected_indices():
    wl = np.array([400.0, 500.0, 700.0])
    values = wl + 1.0  # simple offset for clarity
    targets = np.array([425.0, 575.0, 800.0])

    result = interpolate_values(values, wl, targets, mode="nearest")
    expected = np.array([401.0, 501.0, 701.0])

    np.testing.assert_allclose(result, expected)


def test_interpolate_batch_linear():
    wl = np.array([400.0, 500.0, 700.0])
    batch = np.stack([wl, wl**2])
    targets = np.array([450.0, 600.0])

    result = interpolate_values(batch, wl, targets, mode="linear")

    assert result.shape == (batch.shape[0], targets.shape[0])
    np.testing.assert_allclose(result[0], np.interp(targets, wl, wl))
    np.testing.assert_allclose(result[1], np.interp(targets, wl, wl**2))


def test_resample_by_interpolation_preserves_metadata_and_units():
    wl = np.array([400.0, 500.0, 700.0])
    reflectance = np.array([0.1, 0.2, 0.35])
    spec = Spectrum(
        wavelength_nm=wl,
        values=reflectance,
        kind=QuantityKind.REFLECTANCE,
        units=ValueUnits.REFLECTANCE_FRACTION,
        meta={"source": "test"},
    )

    targets = np.array([450.0, 600.0])
    resampled = resample_by_interpolation(spec, targets, mode="linear")

    np.testing.assert_allclose(resampled.wavelengths.nm, targets)
    np.testing.assert_allclose(resampled.values, np.interp(targets, wl, reflectance))
    assert resampled.kind == spec.kind
    assert resampled.units == spec.units
    assert resampled.meta["source"] == "test"
    assert resampled.meta["resample_mode"] == "center_interp"
    assert resampled.meta["interp_mode"] == "linear"


def test_spline_mode_matches_cubic_spline():
    pytest.importorskip("scipy")
    from scipy.interpolate import CubicSpline

    wl = np.array([0.0, 1.0, 2.0, 3.0])
    values = np.array([0.0, 1.0, 0.0, 1.0])
    targets = np.array([0.5, 1.5, 2.5])

    reference = CubicSpline(wl, values, extrapolate=True)(targets)
    result = interpolate_values(values, wl, targets, mode="spline")

    np.testing.assert_allclose(result, reference)
