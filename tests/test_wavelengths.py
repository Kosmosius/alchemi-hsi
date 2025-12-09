import numpy as np

from alchemi.wavelengths import (
    align_wavelengths,
    check_monotonic,
    ensure_nm,
    fix_monotonic,
    infer_nm,
    to_nm,
    wavelength_equal,
)


def test_to_nm_conversions() -> None:
    microns = np.array([0.4, 0.41], dtype=np.float64)
    angstrom = np.array([4000.0, 4100.0], dtype=np.float64)
    wavenumber = np.array([10_000.0, 9_000.0], dtype=np.float64)

    np.testing.assert_allclose(to_nm(microns, "um"), microns * 1e3)
    np.testing.assert_allclose(to_nm(angstrom, "angstrom"), angstrom * 0.1)
    np.testing.assert_allclose(to_nm(wavenumber, "cm-1"), np.array([1000.0, 1111.11111111]))


def test_infer_and_ensure_nm() -> None:
    nm = np.array([400.0, 401.0], dtype=np.float64)
    microns = np.array([0.4, 0.401], dtype=np.float64)

    np.testing.assert_allclose(infer_nm(nm), nm)
    np.testing.assert_allclose(ensure_nm(microns, None), microns * 1e3)
    np.testing.assert_allclose(ensure_nm(nm, "nanometers"), nm)


def test_check_monotonic_strict_and_non_strict() -> None:
    check_monotonic(np.array([1.0, 2.0, 3.0]))
    try:
        check_monotonic(np.array([1.0, 1.0, 2.0]))
    except ValueError:
        pass
    else:
        raise AssertionError("Expected monotonicity failure for equal entries")

    try:
        check_monotonic(np.array([1.0, 0.95, 1.1]), strict=False, eps=0.1)
    except ValueError:
        raise AssertionError(
            "Non-strict monotonicity should allow small decreases"
        ) from None


def test_fix_monotonic_adjusts_small_violations() -> None:
    corrected = fix_monotonic(np.array([1.0, 1.0, 1.00000000001]))
    check_monotonic(corrected)

    try:
        fix_monotonic(np.array([1.0, 0.5, 1.5]), eps=0.01)
    except ValueError:
        pass
    else:
        raise AssertionError("Large inversions should raise")


def test_wavelength_grid_equality_and_alignment() -> None:
    base = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    perturbed = base + 5e-4
    assert wavelength_equal(base, perturbed, atol=1e-3)
    a, b = align_wavelengths(base, perturbed, atol=1e-3)
    np.testing.assert_allclose(a, b, atol=1e-3)

    far = base + 0.5
    assert not wavelength_equal(base, far)
    try:
        align_wavelengths(base, far)
    except ValueError:
        pass
    else:
        raise AssertionError("Alignment should fail for divergent grids")
