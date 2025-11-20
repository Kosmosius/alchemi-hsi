"""Adapter helpers for SPLIB spectra."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from alchemi.data.io import SPLIBCatalog, load_splib
from alchemi.types import Spectrum

__all__ = ["load_splib_spectrum"]


def load_splib_spectrum(
    src: str | Path | Iterable[str | Path],
    name: str,
    *,
    use_cache: bool = True,
) -> Spectrum:
    """Load the first SPLIB spectrum matching *name* from *src*."""

    catalog = load_splib(src, use_cache=use_cache)
    spectra = _resolve_alias(catalog, name)
    if not spectra:
        raise KeyError(f"No SPLIB spectrum found for {name!r}")
    return spectra[0]


def _resolve_alias(catalog: SPLIBCatalog, name: str) -> list[Spectrum]:
    try:
        return list(catalog.resolve(name))
    except KeyError:
        return list(catalog.get(name, []))
