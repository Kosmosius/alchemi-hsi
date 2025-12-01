"""Temperature–emissivity separation placeholder implementation."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from alchemi.types import Spectrum

__all__ = ["tes_lwirt"]


def tes_lwirt(spectrum: Spectrum, ancillary: dict) -> Tuple[np.ndarray, Spectrum, np.ndarray, np.ndarray]:
    """Perform TES for LWIR radiance spectra.

    Parameters
    ----------
    spectrum:
        Longwave infrared radiance spectrum (typically microns to nm converted).
    ancillary:
        Dictionary of supporting information (e.g. view angles, atmospheric
        profiles). Keys are yet to be finalised.

    Returns
    -------
    Tuple[np.ndarray, Spectrum, np.ndarray, np.ndarray]
        Tuple containing estimated temperature(s), emissivity spectrum,
        uncertainties on temperature, and uncertainties on emissivity.

    Notes
    -----
    This is a placeholder that documents the intended API. A future
    implementation should follow TES literature (e.g., ASTER TES) and provide
    spectral smoothing, NEM/RMSE-based emissivity normalisation, and uncertainty
    propagation.
    """

    raise NotImplementedError("TES retrieval is not yet implemented")
