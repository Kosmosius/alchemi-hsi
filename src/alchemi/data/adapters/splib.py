"""Adapter helpers for SPLIB spectra."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from alchemi.types import Spectrum
from alchemi_hsi.io.splib import load_splib

__all__ = ["load_splib_spectrum"]


def load_splib_spectrum(src: str | Path | Iterable[str | Path], name: str) -> Spectrum:
    """Load a single SPLIB spectrum by name or alias."""

    catalog = load_splib(src)
    spectra = catalog.resolve(name)
    if not spectra:
        raise KeyError(name)
    return spectra[0]
