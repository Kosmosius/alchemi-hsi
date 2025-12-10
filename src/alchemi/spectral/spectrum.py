"""Canonical Spectrum re-export tied to :mod:`alchemi.types`.

Refer to :class:`alchemi.types.Spectrum` for the full Section-4 docstring that
describes wavelength and unit invariants plus factory helpers. This module is a
thin compatibility shim so spectral utilities can import ``Spectrum`` from the
``alchemi.spectral`` namespace.
"""

from alchemi.types import Spectrum

__all__ = ["Spectrum"]
