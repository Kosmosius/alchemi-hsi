"""Utilities for handling wavelength grids and unit conversions.

This module centralises wavelength handling across the codebase. All functions
operate in nanometres internally to align with :class:`alchemi.types.Spectrum`
and :class:`alchemi.types.WavelengthGrid`.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)

_NM_UNITS: set[str] = {"nm", "nanometer", "nanometers", "nanometre", "nanometres"}
_MICRON_UNITS: set[str] = {
    "um",
    "µm",
    "micron",
    "microns",
    "micrometer",
    "micrometers",
    "micrometre",
    "micrometres",
}
_ANGSTROM_UNITS: set[str] = {"angstrom", "angstroms", "å", "a"}
_WAVENUMBER_UNITS: set[str] = {"cm-1", "cm^-1", "wavenumber", "wavenumbers"}

_DEFAULT_FIX_STEP = np.finfo(np.float64).eps

__all__ = [
    "to_nm",
    "infer_nm",
    "ensure_nm",
    "check_monotonic",
    "fix_monotonic",
    "wavelength_equal",
    "align_wavelengths",
]


def _normalize_unit(unit: str | None) -> str | None:
    if unit is None:
        return None
    return unit.strip().lower().replace(" ", "")


def to_nm(values: np.ndarray | Iterable[float], from_units: str | None) -> np.ndarray:
    """Convert wavelength values to nanometres.

    Parameters
    ----------
    values:
        Array-like wavelength values.
    from_units:
        Unit label describing ``values``. Supported inputs include ``nm``
        variants, micrometre spellings (``"um"``, ``"µm"``, ``"micron"`` ...),
        Ångström spellings, and wavenumbers (``"cm-1"``).
    """

    arr = np.asarray(values, dtype=np.float64)
    unit = _normalize_unit(from_units)

    if unit is None or unit in _NM_UNITS:
        return arr.copy()
    if unit in _MICRON_UNITS:
        return arr * 1e3
    if unit in _ANGSTROM_UNITS:
        return arr * 0.1
    if unit in _WAVENUMBER_UNITS:
        return 1.0e7 / arr

    msg = f"Unsupported wavelength units: {from_units!r}"
    raise ValueError(msg)


def infer_nm(values: np.ndarray | Iterable[float]) -> np.ndarray:
    """Infer wavelength units when none are provided.

    The heuristic assumes micrometres when the maximum finite value is below
    ``100`` and nanometres otherwise.
    """

    arr = np.asarray(values, dtype=np.float64)
    try:
        max_val = float(np.nanmax(arr))
    except ValueError:  # empty array
        return arr.copy()
    if max_val < 100.0:
        return arr * 1e3
    return arr.copy()


def ensure_nm(values: np.ndarray | Iterable[float], units: str | None) -> np.ndarray:
    """Convert wavelengths to nanometres using explicit units or heuristics."""

    if units is None:
        return infer_nm(values)
    return to_nm(values, units)


def check_monotonic(wavelength_nm: np.ndarray, *, strict: bool = True, eps: float = 0.0) -> None:
    """Validate that a wavelength grid is monotonic increasing.

    Raises
    ------
    ValueError
        If the array is not (strictly) increasing within the provided
        tolerance ``eps``.
    """

    arr = np.asarray(wavelength_nm, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("Wavelength grid must be a 1-D array")
    if arr.size <= 1:
        return

    diffs = np.diff(arr)
    if strict:
        if np.any(diffs <= eps):
            raise ValueError("Wavelengths must be strictly increasing")
    else:
        if np.any(diffs < -eps):
            raise ValueError("Wavelengths must be non-decreasing")


def fix_monotonic(wavelength_nm: np.ndarray, *, eps: float = 0.0) -> np.ndarray:
    """Return a minimally adjusted strictly-increasing wavelength grid.

    This helper should only be used in salvage workflows; ingestion code should
    prefer :func:`check_monotonic` to fail fast. When small numerical
    violations (``diff <= eps``) are detected the grid is nudged upwards using a
    minimal increment. Larger inversions raise ``ValueError``.
    """

    arr = np.asarray(wavelength_nm, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("Wavelength grid must be a 1-D array")
    if arr.size <= 1:
        return arr.copy()

    margin = max(float(eps), 0.0)
    diffs = np.diff(arr)
    if np.any(diffs < -margin):
        msg = "Monotonicity violations exceed fixable tolerance"
        raise ValueError(msg)

    logger.warning("Applying monotonicity fix; downstream resampling may be preferable.")
    fixed = arr.copy()
    step = max(margin, _DEFAULT_FIX_STEP)
    for idx in range(1, fixed.size):
        min_allowed = fixed[idx - 1] + step
        if fixed[idx] <= min_allowed:
            fixed[idx] = min_allowed
    return fixed


def wavelength_equal(
    a_nm: np.ndarray,
    b_nm: np.ndarray,
    *,
    atol: float = 1e-3,
    rtol: float = 1e-6,
) -> bool:
    """Return ``True`` when two wavelength grids match within tolerance."""

    a_arr = np.asarray(a_nm, dtype=np.float64)
    b_arr = np.asarray(b_nm, dtype=np.float64)
    if a_arr.shape != b_arr.shape:
        return False
    return np.allclose(a_arr, b_arr, atol=atol, rtol=rtol)


def align_wavelengths(
    a_nm: np.ndarray,
    b_nm: np.ndarray,
    *,
    atol: float = 1e-3,
    rtol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Ensure two wavelength grids are aligned without resampling."""

    if wavelength_equal(a_nm, b_nm, atol=atol, rtol=rtol):
        return np.asarray(a_nm, dtype=np.float64), np.asarray(b_nm, dtype=np.float64)

    a_arr = np.asarray(a_nm, dtype=np.float64)
    b_arr = np.asarray(b_nm, dtype=np.float64)
    diff = np.abs(a_arr - b_arr)
    max_diff = float(np.nanmax(diff)) if diff.size else 0.0
    msg = (
        f"Wavelength grids differ beyond tolerance (max abs diff {max_diff} nm, "
        f"atol={atol}, rtol={rtol})"
    )
    raise ValueError(msg)
