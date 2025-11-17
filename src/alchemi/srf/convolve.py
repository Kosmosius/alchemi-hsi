import numpy as np

try:
    from numpy import trapezoid as _integrate
except ImportError:  # pragma: no cover - NumPy < 2.0 fallback
    from numpy import trapz as _integrate  # type: ignore[attr-defined]

from ..types import Spectrum, SpectrumKind, SRFMatrix, WavelengthGrid


def convolve_lab_to_sensor(lab: Spectrum, srf: SRFMatrix) -> Spectrum:
    assert lab.kind == SpectrumKind.REFLECTANCE
    vals = []
    for nm_band, resp in zip(srf.bands_nm, srf.bands_resp, strict=True):
        r = np.interp(nm_band, lab.wavelengths.nm, lab.values)
        vals.append(_integrate(r * resp, nm_band))
    values = np.asarray(vals, dtype=np.float64)
    return Spectrum(
        WavelengthGrid(srf.centers_nm),
        values,
        SpectrumKind.REFLECTANCE,
        "unitless",
        None,
        {"sensor": srf.sensor},
    )
