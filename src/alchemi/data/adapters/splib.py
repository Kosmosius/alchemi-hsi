"""Adapter utilities for working with the SPLIB ingestion pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from alchemi.types import Spectrum
from alchemi_hsi.io.splib import SPLIBCatalog, load_splib


def load_splib_spectrum(path: str | Path, name: str, *, use_cache: bool = True) -> Spectrum:
    """Load the first SPLIB spectrum matching *name* from *path*."""

    catalog = load_splib(path, use_cache=use_cache)
    spectra = _resolve_alias(catalog, name)
    if not spectra:
        raise KeyError(f"No SPLIB spectrum found for {name!r}")
    return spectra[0]


def _resolve_alias(catalog: SPLIBCatalog, name: str) -> Iterable[Spectrum]:
    try:
        return catalog.resolve(name)
    except KeyError:
        return catalog.get(name, [])
