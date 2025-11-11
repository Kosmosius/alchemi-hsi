import numpy as np


def inject_synthetic_plume(img: np.ndarray, mask: np.ndarray, strength: float = 0.02) -> np.ndarray:
    """
    img: [H,W,B] reflectance or radiance; mask: [H,W] plume binary
    Apply multiplicative absorption anomaly at bands 2..5 for toy test.
    """
    out = img.copy()
    bands = slice(2, 6)
    out[mask, bands] *= 1.0 - strength
    return out
