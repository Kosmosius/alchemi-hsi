import numpy as np

h = 6.62607015e-34
c = 2.99792458e8
k = 1.380649e-23


def bt_to_radiance(bt_K: np.ndarray, wavelength_nm: np.ndarray) -> np.ndarray:
    lam = wavelength_nm * 1e-9
    val = (2 * h * c**2) / (lam**5) / (np.expm1((h * c) / (lam * k * bt_K)))
    return val * 1e-9


def radiance_to_bt(L: np.ndarray, wavelength_nm: np.ndarray) -> np.ndarray:
    lam = wavelength_nm * 1e-9
    L_m = L * 1e9
    inside = 1.0 + (2 * h * c**2) / (L_m * lam**5)
    return (h * c) / (lam * k * np.log(inside))
