import numpy as np

try:
    from numpy import trapezoid as _integrate
except ImportError:  # pragma: no cover - NumPy < 2.0 fallback
    from numpy import trapz as _integrate  # type: ignore[attr-defined]

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
    M, _ = lab_values.shape
    B = len(srf.centers_nm)
    out = np.zeros((M, B), dtype=np.float64)
    for b, (nm, resp) in enumerate(zip(srf.bands_nm, srf.bands_resp, strict=False)):
        interpolated = np.vstack([np.interp(nm, lab_nm, sample) for sample in lab_values])
        out[:, b] = _integrate(interpolated * resp[None, :], nm, axis=1)
    return out
