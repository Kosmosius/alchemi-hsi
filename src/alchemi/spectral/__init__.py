"""Spectral data and metadata model.

The types exported here mirror the contracts described in the design doc's
"Data and metadata model" section, covering canonical spectra, SRFs, and
per-pixel samples.
"""

from .sample import BandMetadata, GeoMeta, Sample, ViewingGeometry
from .spectrum import Spectrum
from .srf import SRFMatrix

__all__ = [
    "BandMetadata",
    "GeoMeta",
    "Sample",
    "SRFMatrix",
    "Spectrum",
    "ViewingGeometry",
]
