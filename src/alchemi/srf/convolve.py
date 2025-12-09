import numpy as np

from ..types import Spectrum, SpectrumKind, SRFMatrix, WavelengthGrid
from .resample import resample_values_with_srf
from .sensor import SensorSRF


def convolve_lab_to_sensor(lab: Spectrum, srf: SRFMatrix) -> Spectrum:
    assert lab.kind == SpectrumKind.REFLECTANCE
    srfs = np.vstack(
        [
            np.interp(lab.wavelengths.nm, nm_band, resp, left=0.0, right=0.0)
            for nm_band, resp in zip(srf.bands_nm, srf.bands_resp, strict=True)
        ]
    )
    sensor_srf = SensorSRF(
        wavelength_grid_nm=lab.wavelengths.nm,
        srfs=srfs,
        band_centers_nm=np.asarray(srf.centers_nm, dtype=np.float64),
        meta={"sensor": srf.sensor},
    )
    values, _ = resample_values_with_srf(lab.values, lab.wavelengths.nm, sensor_srf)
    return Spectrum(
        WavelengthGrid(
            sensor_srf.band_centers_nm if sensor_srf.band_centers_nm is not None else srf.centers_nm
        ),
        values,
        SpectrumKind.REFLECTANCE,
        "unitless",
        None,
        {"sensor": srf.sensor},
    )
