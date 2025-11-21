"""Loader for SpectralEarth-style spectral samples.

This helper enforces nanometre spectral units and keeps track of whether
spectra are radiance or reflectance, following the canonical DATA_SPEC
conventions used throughout Alchemi.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from alchemi.types import Spectrum, SpectrumKind, WavelengthGrid

__all__ = ["load_spectralearth"]

# Default radiance units in the Alchemi DATA_SPEC.
_RAD_UNITS = "W·m⁻²·sr⁻¹·nm⁻¹"

# Warn about wavelength unit conversion at most once to avoid noisy logs.
_WARNED_WAVELENGTH_UNITS = False


def load_spectralearth(sample: str | Path | Mapping[str, object]) -> Spectrum:
    """Load a single SpectralEarth sample into a :class:`~alchemi.types.Spectrum`.

    Parameters
    ----------
    sample:
        Either a mapping describing the spectral sample or a path to a ``.npz``
        archive containing a 1-D wavelength axis and either ``radiance`` or
        ``reflectance`` arrays. Optional keys describing wavelength units and
        band validity masks are honoured when present.

    Returns
    -------
    Spectrum
        Normalised spectrum with a strictly increasing nanometre wavelength
        grid, units consistent with the spectral quantity, and an optional
        propagated band mask.
    """
    payload: MutableMapping[str, object]
    if isinstance(sample, (str, Path)):
        path = Path(sample)
        with np.load(path) as data:
            payload = {k: data[k].copy() for k in data.files}
    elif isinstance(sample, Mapping):
        # Copy so we can mutate locally without surprising the caller.
        payload = dict(sample)
    else:  # pragma: no cover - defensive
        raise TypeError("sample must be a mapping or path to an .npz archive")

    wavelengths = _extract_wavelengths(payload)

    radiance = payload.get("radiance")
    reflectance = payload.get("reflectance")
    if (radiance is None) == (reflectance is None):
        msg = "Exactly one of 'radiance' or 'reflectance' must be present"
        raise ValueError(msg)

    if radiance is not None:
        values = np.asarray(radiance, dtype=np.float64)
        kind = SpectrumKind.RADIANCE
        units = str(payload.get("radiance_units") or _RAD_UNITS)
    else:
        values = np.asarray(reflectance, dtype=np.float64)
        kind = SpectrumKind.REFLECTANCE
        units = "1"

    if values.ndim != 1:
        msg = "Spectral values must be a 1-D array aligned with the wavelength grid"
        raise ValueError(msg)
    if values.shape != wavelengths.shape:
        msg = "Spectral values must have the same length as the wavelength grid"
        raise ValueError(msg)

    band_valid_mask = payload.get("band_valid_mask")
    mask = None
    if band_valid_mask is not None:
        mask_arr = np.asarray(band_valid_mask, dtype=bool)
        if mask_arr.shape != wavelengths.shape:
            msg = "band_valid_mask must match wavelength grid length"
            raise ValueError(msg)
        mask = mask_arr

    spectrum = Spectrum(
        wavelengths=WavelengthGrid(wavelengths),
        values=values,
        kind=kind,
        units=units,
        mask=mask,
        meta={"quantity": kind.value},
    )
    return spectrum


def _extract_wavelengths(payload: Mapping[str, object]) -> NDArray[np.float64]:
    """Locate and normalise the wavelength axis in *payload*."""

    # Prefer explicit / canonical field names.
    candidates: Sequence[str] = (
        "wavelengths_nm",
        "wavelength_nm",
        "wavelengths",
        "wavelength",
    )

    wavelengths: NDArray[np.float64] | None = None
    unit: str | None = None

    for key in candidates:
        if key in payload:
            wavelengths = np.asarray(payload[key], dtype=np.float64)
            unit = _guess_unit(key, payload)
            break

    if wavelengths is None:
        # Fall back to a looser name match for more forgiving inputs,
        # similar in spirit to Option B's coordinate matching.
        for key, value in payload.items():
            name = str(key).lower()
            if any(token in name for token in ("wave", "lambda", "spectral")):
                wavelengths = np.asarray(value, dtype=np.float64)
                unit = _guess_unit(str(key), payload)
                break

    if wavelengths is None:
        raise ValueError("No wavelength field found in sample")

    return _ensure_nanometers(wavelengths, unit)


def _guess_unit(key: str, payload: Mapping[str, object]) -> str | None:
    """Infer a unit string for the wavelength axis from common key patterns."""

    # An explicit 'unit' key wins.
    unit = payload.get("unit")
    if unit is not None:
        return str(unit)

    # Try <key>_unit / <key>_units first.
    for suffix in ("unit", "units"):
        candidate = f"{key}_{suffix}"
        if candidate in payload:
            return str(payload[candidate])

    # Then fall back to wavelength_unit / wavelength_units variants.
    for base in ("wavelength", "wavelengths"):
        for suffix in ("unit", "units"):
            candidate = f"{base}_{suffix}"
            if candidate in payload:
                return str(payload[candidate])

    return None


def _ensure_nanometers(wavelengths: np.ndarray, unit: str | None) -> NDArray[np.float64]:
    """Return a 1-D, strictly increasing nanometre wavelength grid."""
    global _WARNED_WAVELENGTH_UNITS

    arr: NDArray[np.float64] = np.asarray(wavelengths, dtype=np.float64)

    unit_norm = ""
    if isinstance(unit, str):
        # Normalise units string: lower-case and strip spaces.
        unit_norm = unit.strip().lower().replace(" ", "")

    convert = False
    if unit_norm:
        if any(token in unit_norm for token in ("µm", "um", "microm", "micron")):
            convert = True
        elif "nm" in unit_norm or "nanom" in unit_norm:
            convert = False
        else:
            msg = f"Unsupported wavelength units: {unit}"
            raise ValueError(msg)
    else:
        # No explicit units: treat "small" axes as micrometres, otherwise leave
        # as-is and assume the caller already provided nanometres.
        if np.nanmax(arr) <= 20.0:
            convert = True

    if convert:
        arr = arr * 1000.0
        if not _WARNED_WAVELENGTH_UNITS:
            warnings.warn(
                "Converting wavelengths from micrometre (μm) units to nanometres",
                RuntimeWarning,
                stacklevel=2,
            )
            _WARNED_WAVELENGTH_UNITS = True

    if arr.ndim != 1 or np.any(np.diff(arr) <= 0):
        raise ValueError("Wavelengths must be a strictly increasing 1-D array in nanometres")

    return arr
