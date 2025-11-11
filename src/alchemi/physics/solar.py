import numpy as np


# Minimal placeholder ASTM-like E0 spectrum sampler.
# Replace with tabulated ASTM G-173 or mission-provided E0 as needed.
def get_E0_nm(wavelength_nm: np.ndarray) -> np.ndarray:
    # crude smooth curve peaking in visible, decaying in SWIR
    nm = wavelength_nm.astype(np.float64)
    peak = 550.0
    width = 300.0
    base = np.exp(-0.5 * ((nm - peak) / width) ** 2) * 1.9
    tail = 0.4 * np.exp(-(nm - 1000.0) / 800.0)
    return (base + tail).clip(min=1e-6)


def sun_earth_factor(doy: int) -> float:
    theta = np.deg2rad(0.9856 * (doy - 4))
    R = 1.0 - 0.01672 * np.cos(theta)
    return 1.0 / (R * R + 1e-9)
