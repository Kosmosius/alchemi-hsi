import numpy as np
from numpy.typing import NDArray

from ..types import SRFMatrix
from ..utils.integrate import np_integrate as _np_integrate


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
    M, _ = lab_values.shape
    B = len(srf.centers_nm)
    out = np.zeros((M, B), dtype=np.float64)
    for b, (nm, resp) in enumerate(zip(srf.bands_nm, srf.bands_resp, strict=True)):
        interpolated = np.vstack([np.interp(nm, lab_nm, sample) for sample in lab_values])
        out[:, b] = _np_integrate(interpolated * resp[None, :], nm, axis=1)
    return np.asarray(out, dtype=np.float64)
