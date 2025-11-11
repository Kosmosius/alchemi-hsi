import numpy as np

from ..types import Spectrum, SpectrumKind, WavelengthGrid, SRFMatrix


def convolve_lab_to_sensor(lab: Spectrum, srf: SRFMatrix) -> Spectrum:
    assert lab.kind == SpectrumKind.REFLECTANCE
    vals = []
    for nm_band, resp in zip(srf.bands_nm, srf.bands_resp):
        r = np.interp(nm_band, lab.wavelengths.nm, lab.values)
        vals.append(np.trapz(r * resp, nm_band))
    values = np.asarray(vals, dtype=np.float64)
    return Spectrum(
        WavelengthGrid(srf.centers_nm),
        values,
        SpectrumKind.REFLECTANCE,
        "unitless",
        None,
        {"sensor": srf.sensor},
    )
