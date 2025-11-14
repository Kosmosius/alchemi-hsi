"""Property-based checks for SRF resampling utilities."""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from alchemi.srf import resample
from alchemi.types import SRFMatrix


@st.composite
def _random_srf(draw) -> SRFMatrix:
    n_bands = draw(st.integers(min_value=1, max_value=3))
    centers: list[float] = []
    bands_nm: list[np.ndarray] = []
    bands_resp: list[np.ndarray] = []
    for _ in range(n_bands):
        n_points = draw(st.integers(min_value=4, max_value=20))
        start = draw(st.floats(min_value=400.0, max_value=2500.0))
        increments = draw(
            st.lists(
                st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
                min_size=n_points - 1,
                max_size=n_points - 1,
            )
        )
        wl = np.empty(n_points, dtype=np.float64)
        wl[0] = start
        wl[1:] = start + np.cumsum(np.asarray(increments, dtype=np.float64))
        resp = draw(
            st.lists(
                st.floats(min_value=1e-6, max_value=5.0, allow_nan=False, allow_infinity=False),
                min_size=n_points,
                max_size=n_points,
            )
        )
        resp_arr = np.asarray(resp, dtype=np.float64)
        bands_nm.append(wl)
        bands_resp.append(resp_arr)
        centers.append(draw(st.floats(min_value=float(wl[0]), max_value=float(wl[-1]))))

    return SRFMatrix(
        sensor="test",
        centers_nm=np.asarray(centers, dtype=np.float64),
        bands_nm=bands_nm,
        bands_resp=bands_resp,
    )


@st.composite
def _gaussian_case(draw):
    center = draw(st.floats(min_value=500.0, max_value=2500.0))
    fwhm = draw(st.floats(min_value=5.0, max_value=80.0))
    span = 4.0 * fwhm
    n_points = draw(st.integers(min_value=200, max_value=500))
    wl = np.linspace(center - span, center + span, n_points, dtype=np.float64)
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    weights = np.exp(-0.5 * ((wl - center) / sigma) ** 2)

    base_terms = np.vstack((np.ones_like(wl), wl - center, (wl - center) ** 2))
    n_spectra = draw(st.integers(min_value=1, max_value=3))
    spectra = []
    for _ in range(n_spectra):
        coeffs = np.asarray(
            draw(
                st.lists(
                    st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
                    min_size=3,
                    max_size=3,
                )
            ),
            dtype=np.float64,
        )
        spectrum = coeffs[:, None] * base_terms
        spectra.append(np.sum(spectrum, axis=0))

    spectra_arr = np.vstack(spectra)
    srf = SRFMatrix(
        sensor="gaussian",
        centers_nm=np.asarray([center], dtype=np.float64),
        bands_nm=[wl],
        bands_resp=[weights],
    ).normalize_trapz()

    return wl, spectra_arr, center, fwhm, srf


@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_random_srf())
def test_srf_normalization_is_close(random_srf: SRFMatrix) -> None:
    """SRF normalization should yield unit integrals per band."""

    normalized = random_srf.normalize_trapz()
    integrals = normalized.row_integrals()
    np.testing.assert_allclose(integrals, np.ones_like(integrals), rtol=1e-10, atol=1e-10)


@settings(max_examples=30, deadline=None)
@given(_gaussian_case())
def test_gaussian_matches_tabulated(case) -> None:
    """Gaussian analytic kernels should match trapezoid integration of tabulated SRFs."""

    wl, spectra, center, fwhm, srf = case
    centers = np.asarray([center], dtype=np.float64)
    expected = resample.convolve_to_bands(wl, spectra, srf)
    actual = resample.gaussian_resample(wl, spectra, centers, np.asarray([fwhm], dtype=np.float64))
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-8)
