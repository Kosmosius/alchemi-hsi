"""Spectral data and metadata model.

The types exported here mirror the contracts described in the design doc's
"Data and metadata model" section, covering canonical spectra, SRFs, and
per-pixel samples. ``Spectrum`` is a thin re-export of
``alchemi.types.Spectrum`` to preserve backward compatibility while using a
single canonical implementation.
"""

from .sample import BandMetadata, GeoMeta, Sample, ViewingGeometry
from .spectrum import Spectrum
from .srf import SRFMatrix
from .validate import validate_sample, validate_spectrum_physics

__all__ = [
    "BandMetadata",
    "GeoMeta",
    "Sample",
    "SRFMatrix",
    "Spectrum",
    "ViewingGeometry",
    "validate_sample",
    "validate_spectrum_physics",
]
