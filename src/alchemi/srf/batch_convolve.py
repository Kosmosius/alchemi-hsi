import numpy as np

from ..types import SRFMatrix


def batch_convolve_lab_to_sensor(
    lab_nm: np.ndarray, lab_values: np.ndarray, srf: SRFMatrix
) -> np.ndarray:
    """
    lab_nm: [N] grid (same for batch)
    lab_values: [M, N] batch of lab reflectance
    returns: [M, B_sensor]
    """
    lab_nm = np.asarray(lab_nm, dtype=np.float64)
    lab_values = np.asarray(lab_values, dtype=np.float64)
    M, N = lab_values.shape
    B = len(srf.centers_nm)
    out = np.zeros((M, B), dtype=np.float64)
    for b, (nm, resp) in enumerate(zip(srf.bands_nm, srf.bands_resp)):
        interpolated = np.vstack([np.interp(nm, lab_nm, sample) for sample in lab_values])
        out[:, b] = np.trapz(interpolated * resp[None, :], nm, axis=1)
    return out
