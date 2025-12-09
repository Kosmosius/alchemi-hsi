import numpy as np
from numpy.typing import NDArray

from ..types import SRFMatrix
from .resample import resample_values_with_srf
from .sensor import SensorSRF


def batch_convolve_lab_to_sensor(
    lab_nm: np.ndarray, lab_values: np.ndarray, srf: SRFMatrix
) -> NDArray[np.float64]:
    """
    lab_nm: [N] grid (same for batch)
    lab_values: [M, N] batch of lab reflectance
    returns: [M, B_sensor]
    """
    lab_nm = np.asarray(lab_nm, dtype=np.float64)
    lab_values = np.asarray(lab_values, dtype=np.float64)
    srfs = np.vstack(
        [
            np.interp(lab_nm, nm, resp, left=0.0, right=0.0)
            for nm, resp in zip(srf.bands_nm, srf.bands_resp, strict=True)
        ]
    )
    sensor_srf = SensorSRF(
        wavelength_grid_nm=lab_nm,
        srfs=srfs,
        band_centers_nm=np.asarray(srf.centers_nm, dtype=np.float64),
        meta={"sensor": getattr(srf, "sensor", None)},
    ).normalized()
    band_values, _ = resample_values_with_srf(lab_values, lab_nm, sensor_srf)
    return np.asarray(band_values, dtype=np.float64)
