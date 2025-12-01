from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..types import QuantityKind, Spectrum, ValueUnits, WavelengthGrid

_DEFAULT_LIBRARY_ROOT = Path("resources/examples/tiny_lab_library_subset")


@dataclass
class LibraryEntry:
    entry_id: str
    ontology_leaf_id: str
    spectrum: Spectrum
    meta: dict[str, Any] = field(default_factory=dict)


_LIBRARY_CACHE: list[LibraryEntry] | None = None


def _load_entry(path: Path) -> LibraryEntry:
    data = json.loads(path.read_text())
    wavelengths = np.asarray(data["wavelengths_nm"], dtype=np.float64)
    values = np.asarray(data["reflectance"], dtype=np.float64)
    spectrum = Spectrum(
        wavelengths=WavelengthGrid(wavelengths),
        values=values,
        kind=QuantityKind.REFLECTANCE,
        units=ValueUnits.REFLECTANCE,
        meta=data.get("meta", {}),
    )
    return LibraryEntry(
        entry_id=data["entry_id"],
        ontology_leaf_id=data["ontology_leaf_id"],
        spectrum=spectrum,
        meta=data.get("meta", {}),
    )


def load_library_subset(root: str | Path | None = None) -> list[LibraryEntry]:
    """Load a subset of lab spectra from JSON files.

    The default location ships with a tiny set of placeholder spectra to keep
    the registry lightweight. TODO: expand to the full SPLIB v7 distribution
    when available.
    """

    base = Path(root) if root is not None else _DEFAULT_LIBRARY_ROOT
    if not base.exists():
        raise FileNotFoundError(f"Library path not found: {base}")

    entries: list[LibraryEntry] = []
    for path in sorted(base.glob("*.json")):
        entries.append(_load_entry(path))

    global _LIBRARY_CACHE
    _LIBRARY_CACHE = entries
    return entries


def _ensure_cache() -> list[LibraryEntry]:
    global _LIBRARY_CACHE
    if _LIBRARY_CACHE is None:
        _LIBRARY_CACHE = load_library_subset()
    return _LIBRARY_CACHE


def get_entries_for_leaf(leaf_id: str) -> list[LibraryEntry]:
    return [entry for entry in _ensure_cache() if entry.ontology_leaf_id == leaf_id]


def list_all_entries() -> list[LibraryEntry]:
    return list(_ensure_cache())


__all__ = ["LibraryEntry", "load_library_subset", "get_entries_for_leaf", "list_all_entries"]
