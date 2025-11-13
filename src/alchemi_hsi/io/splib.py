"""Loader for the USGS Spectral Library (SPLIB)."""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Iterable, Iterator, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from alchemi.types import Spectrum, SpectrumKind, WavelengthGrid

__all__ = ["SPLIBCatalog", "load_splib"]


@dataclass
class _RawEntry:
    name: str
    wavelengths_nm: np.ndarray
    reflectance: np.ndarray
    meta: dict[str, object]
    source: Path


class SPLIBCatalog(dict[str, list[Spectrum]]):
    """Dictionary mapping canonical material names to spectra with alias utilities."""

    alias_map: dict[str, str]
    aliases: dict[str, list[str]]

    def __init__(
        self,
        data: Mapping[str, Sequence[Spectrum]] | None = None,
        *,
        alias_map: Mapping[str, str] | None = None,
        aliases: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        super().__init__()
        if data:
            for name, spectra in data.items():
                self[name.lower()] = list(spectra)
        self.alias_map = {k.lower(): v.lower() for k, v in (alias_map or {}).items()}
        self.aliases = {k.lower(): sorted(v) for k, v in (aliases or {}).items()}

    def canonical_name(self, name: str) -> str:
        key = _normalize_name(name)
        return self.alias_map.get(key, key)

    def resolve(self, name: str) -> list[Spectrum]:
        canonical = self.canonical_name(name)
        if canonical not in self:
            raise KeyError(name)
        return self[canonical]

    def add_spectrum(self, name: str, spectrum: Spectrum) -> None:
        canonical = self.canonical_name(name)
        self.setdefault(canonical, []).append(spectrum)

    def register_alias(self, alias: str, canonical: str) -> None:
        alias_key = _normalize_name(alias)
        canonical_key = _normalize_name(canonical)
        self.alias_map[alias_key] = canonical_key
        self.aliases.setdefault(canonical_key, []).append(alias)
        self.aliases[canonical_key] = sorted(set(self.aliases[canonical_key]))


def load_splib(src: str | os.PathLike[str] | Iterable[str | os.PathLike[str]]) -> SPLIBCatalog:
    """Parse SPLIB ASCII files into a :class:`SPLIBCatalog`."""

    files = _gather_files(src)
    entries = []
    for file in files:
        entries.extend(_parse_file(file))
    if not entries:
        raise ValueError(f"No SPLIB spectra found in {src}")

    alias_lookup: dict[str, str] = {}
    aliases_by_canonical: MutableMapping[str, set[str]] = {}
    catalog_data: dict[str, list[Spectrum]] = {}

    for entry in entries:
        canonical = _normalize_name(entry.name)
        spectrum = Spectrum(
            wavelengths=WavelengthGrid(entry.wavelengths_nm),
            values=entry.reflectance,
            kind=SpectrumKind.REFLECTANCE,
            units="reflectance",
            mask=None,
            meta={"source": str(entry.source), **entry.meta},
        )
        catalog_data.setdefault(canonical, []).append(spectrum)

        for alias in _collect_aliases(entry.meta):
            alias_lookup[_normalize_name(alias)] = canonical
            aliases_by_canonical.setdefault(canonical, set()).add(alias)

    aliases_final = {k: sorted(v) for k, v in aliases_by_canonical.items()}
    catalog = SPLIBCatalog(catalog_data, alias_map=alias_lookup, aliases=aliases_final)
    return catalog


def _gather_files(src: str | os.PathLike[str] | Iterable[str | os.PathLike[str]]) -> list[Path]:
    if isinstance(src, (str, os.PathLike)):
        paths = [Path(src)]
    else:
        paths = [Path(p) for p in src]
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.glob("**/*.txt")))
        elif path.is_file():
            files.append(path)
    return files


def _parse_file(file: Path) -> Iterator[_RawEntry]:
    content = file.read_text(encoding="utf-8", errors="replace").splitlines()
    metadata: dict[str, str] = {}
    data_lines: list[str] = []
    name = file.stem
    for line in content:
        stripped = line.strip()
        if not stripped:
            if metadata or data_lines:
                yield _entry_from_metadata(name, metadata, data_lines, file)
                metadata = {}
                data_lines = []
            continue
        if ":" in stripped:
            key, value = stripped.split(":", 1)
            metadata[key.strip().lower()] = value.strip()
            if key.strip().lower() in {"name", "material", "target"}:
                name = value.strip()
        else:
            data_lines.append(stripped)
    if metadata or data_lines:
        yield _entry_from_metadata(name, metadata, data_lines, file)


def _entry_from_metadata(
    name: str, metadata: Mapping[str, str], data_lines: Sequence[str], source: Path
) -> _RawEntry:
    array = _parse_data_lines(data_lines, metadata)
    wavelengths = array[:, 0]
    reflectance = array[:, 1]
    units = metadata.get("units", "nm")
    wavelengths_nm = _convert_wavelengths(wavelengths, units)
    meta: dict[str, object] = dict(metadata)
    meta.setdefault("hash", hashlib.sha1(reflectance.tobytes()).hexdigest())
    return _RawEntry(
        name=name,
        wavelengths_nm=wavelengths_nm,
        reflectance=reflectance,
        meta=meta,
        source=source,
    )


def _parse_data_lines(data_lines: Sequence[str], metadata: Mapping[str, str]) -> np.ndarray:
    rows = []
    for line in data_lines:
        parts = _split_columns(line)
        if len(parts) < 2:
            continue
        try:
            rows.append([float(parts[0]), float(parts[1])])
        except ValueError:
            continue
    if not rows:
        raise ValueError("No spectral samples found in SPLIB entry")
    return np.asarray(rows, dtype=np.float64)


def _split_columns(line: str) -> list[str]:
    if "," in line:
        parts = [part.strip() for part in line.split(",")]
    else:
        parts = re.split(r"\s+", line.strip())
    return [p for p in parts if p]


def _convert_wavelengths(values: np.ndarray, units: str) -> np.ndarray:
    units = (units or "").replace("μ", "u").lower()
    if units in {"um", "micrometer", "micrometers", "micron", "microns"}:
        return values * 1000.0
    if units in {"m"}:
        return values * 1e9
    if units in {"nm", "nanometer", "nanometers", "nanometre"}:
        return values
    return values


def _collect_aliases(meta: Mapping[str, object]) -> list[str]:
    aliases = []
    for key in ("alias", "aliases", "common", "synonyms"):
        value = meta.get(key)
        if not value:
            continue
        if isinstance(value, str):
            parts = re.split(r"[,;]", value)
        else:
            parts = list(value)
        aliases.extend([part.strip() for part in parts if part.strip()])
    return aliases


def _normalize_name(name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
    return re.sub(r"\s+", " ", cleaned)
