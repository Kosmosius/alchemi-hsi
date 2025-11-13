"""Loader for the USGS Spectral Library (SPLIB)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Sequence

import hashlib
import os
import re

import numpy as np

from alchemi.types import Spectrum, SpectrumKind, WavelengthGrid

__all__ = ["load_splib", "SPLIBCatalog"]


@dataclass
class _RawEntry:
    canonical: str
    alias_tokens: Sequence[str]
    wavelengths_nm: np.ndarray
    reflectance: np.ndarray
    meta: Dict[str, object]
    source: Path


class SPLIBCatalog(dict[str, List[Spectrum]]):
    """Dictionary mapping canonical material names to spectra with alias utilities."""

    alias_map: Dict[str, str]
    aliases: Dict[str, List[str]]

    def __init__(
        self,
        *args,
        alias_map: Mapping[str, str] | None = None,
        aliases: Mapping[str, Sequence[str]] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.alias_map = dict(alias_map or {})
        self.aliases = {k: sorted(v) for k, v in (aliases or {}).items()}

    def resolve(self, name: str) -> List[Spectrum]:
        canonical = self.canonical_name(name)
        return self[canonical]

    def canonical_name(self, name: str) -> str:
        key = _normalize_key(name)
        try:
            return self.alias_map[key]
        except KeyError as exc:
            raise KeyError(name) from exc


_CACHE_SUFFIX = ".splib-cache.npz"
_DATA_EXTENSIONS = {".txt", ".csv", ".asc", ".dat", ".tsv"}
_METADATA_KEYS = {
    "name",
    "material",
    "material_id",
    "sample",
    "id",
    "canonical_name",
}
_ALIAS_KEYS = {"alias", "aliases", "synonym", "synonyms"}
_WAVELENGTH_KEYS = {
    "wavelength_unit",
    "wavelength_units",
    "wavelengths_unit",
    "wavelengths_units",
    "wavelength_units_nm",
}
_REFLECTANCE_KEYS = {
    "reflectance_unit",
    "reflectance_units",
    "units",
    "quantity",
}


def load_splib(path: str | os.PathLike[str], *, use_cache: bool = True) -> SPLIBCatalog:
    """Parse SPLIB spectra from *path*.

    The loader accepts individual text/CSV files or directories containing
    multiple spectra files. Spectral tables are expected to contain wavelength
    and reflectance columns. Wavelengths are normalised to nanometres and
    reflectances to fractional units in the [0, 1] range.
    """

    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"SPLIB source not found: {path}")

    src_hash = _hash_source(src)
    cache_path = _cache_path(src)

    if use_cache and cache_path.exists():
        try:
            catalog = _load_cache(cache_path, src_hash)
            if catalog is not None:
                return catalog
        except Exception:
            # Ignore cache failures and rebuild from source
            pass

    entries = list(_parse_entries(src))
    if not entries:
        raise ValueError(f"No SPLIB spectra found in {src}")

    alias_lookup: Dict[str, str] = {}
    aliases_by_canonical: MutableMapping[str, set[str]] = {}
    for entry in entries:
        aliases = aliases_by_canonical.setdefault(entry.canonical, set())
        for token in entry.alias_tokens:
            if not token:
                continue
            clean = _clean_name(token)
            if not clean:
                continue
            aliases.add(clean)
            norm = _normalize_key(clean)
            alias_lookup.setdefault(norm, entry.canonical)
        aliases.add(entry.canonical)
        alias_lookup.setdefault(_normalize_key(entry.canonical), entry.canonical)

    catalog = SPLIBCatalog(alias_map=alias_lookup, aliases=aliases_by_canonical)
    for entry in entries:
        spectrum = Spectrum(
            wavelengths=WavelengthGrid(np.asarray(entry.wavelengths_nm, dtype=np.float64)),
            values=np.asarray(entry.reflectance, dtype=np.float64),
            kind=SpectrumKind.REFLECTANCE,
            units="unitless",
            meta={
                **entry.meta,
                "sensor": "LAB",
                "quantity": "reflectance",
                "canonical_name": entry.canonical,
                "aliases": sorted(aliases_by_canonical[entry.canonical]),
                "source": str(entry.source),
            },
        )
        catalog.setdefault(entry.canonical, []).append(spectrum)

    if use_cache:
        _write_cache(cache_path, src_hash, catalog)

    return catalog


def _parse_entries(path: Path) -> Iterator[_RawEntry]:
    if path.is_file():
        files = [path]
    else:
        files = [
            p
            for p in sorted(path.rglob("*"))
            if p.is_file() and p.suffix.lower() in _DATA_EXTENSIONS
        ]
    for file in files:
        yield from _parse_file(file)


def _parse_file(file: Path) -> Iterator[_RawEntry]:
    content = file.read_text(encoding="utf-8", errors="replace").splitlines()
    metadata: Dict[str, str] = {}
    data_lines: List[str] = []
    in_data = False
    for line in content:
        stripped = line.strip()
        if not stripped:
            if in_data:
                data_lines.append(stripped)
            continue
        if not in_data and stripped.startswith("#"):
            key, value = _parse_metadata_line(stripped[1:].strip())
            if key:
                metadata[key] = value
            continue
        tokens = _split_columns(stripped)
        if not tokens:
            continue
        if not in_data and not _tokens_are_numeric(tokens):
            # header row, inspect for inline metadata
            _ingest_header_metadata(tokens, metadata)
            continue
        in_data = True
        data_lines.append(stripped)
    if not data_lines:
        return
    table = _parse_numeric_table(data_lines)
    wavelengths = table[:, 0]
    reflectance = table[:, 1]

    wl_unit = _detect_unit(metadata, _WAVELENGTH_KEYS)
    ref_unit = _detect_unit(metadata, _REFLECTANCE_KEYS)

    wavelengths_nm = _normalize_wavelengths(wavelengths, wl_unit)
    reflectance = _normalize_reflectance(reflectance, ref_unit)

    canonical = _infer_canonical_name(metadata, file)
    alias_tokens = list(_collect_aliases(metadata, file))

    yield _RawEntry(
        canonical=canonical,
        alias_tokens=alias_tokens,
        wavelengths_nm=wavelengths_nm,
        reflectance=reflectance,
        meta={k: v for k, v in metadata.items() if k not in _ALIAS_KEYS},
        source=file,
    )


def _parse_metadata_line(line: str) -> tuple[str | None, str]:
    if not line:
        return None, ""
    match = re.match(r"([^:=]+)[:=](.*)", line)
    if match:
        key = match.group(1).strip().lower()
        value = match.group(2).strip()
        return key, value
    return line.strip().lower(), ""


def _split_columns(line: str) -> List[str]:
    if "," in line:
        parts = [part.strip() for part in line.split(",")]
    else:
        parts = [part for part in re.split(r"\s+", line) if part]
    return parts


def _tokens_are_numeric(tokens: Sequence[str]) -> bool:
    for token in tokens:
        try:
            float(token)
        except ValueError:
            return False
    return True


def _ingest_header_metadata(tokens: Sequence[str], metadata: MutableMapping[str, str]) -> None:
    joined = " ".join(tokens)
    lowered = joined.lower()
    if "micron" in lowered or "um" in lowered:
        metadata.setdefault("wavelength_unit", "micron")
    elif "nanometer" in lowered or "nm" in lowered:
        metadata.setdefault("wavelength_unit", "nm")
    if "percent" in lowered or "%" in joined:
        metadata.setdefault("reflectance_unit", "percent")


def _parse_numeric_table(lines: Sequence[str]) -> np.ndarray:
    data = []
    for line in lines:
        if not line:
            continue
        tokens = _split_columns(line)
        if len(tokens) < 2:
            continue
        try:
            row = [float(tokens[0]), float(tokens[1])]
        except ValueError as exc:
            raise ValueError(f"Non-numeric data row in {lines}") from exc
        data.append(row)
    if not data:
        raise ValueError("SPLIB spectrum missing numeric data")
    arr = np.asarray(data, dtype=np.float64)
    arr = arr[np.argsort(arr[:, 0])]
    arr = _deduplicate_wavelengths(arr)
    return arr


def _deduplicate_wavelengths(arr: np.ndarray) -> np.ndarray:
    wavelengths, indices = np.unique(arr[:, 0], return_index=True)
    values = arr[indices, 1]
    order = np.argsort(wavelengths)
    wavelengths = wavelengths[order]
    values = values[order]
    return np.column_stack([wavelengths, values])


def _detect_unit(metadata: Mapping[str, str], keys: Iterable[str]) -> str | None:
    for key in keys:
        value = metadata.get(key)
        if value:
            return value
    return None


def _normalize_wavelengths(values: np.ndarray, unit: str | None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    unit_norm = (unit or "").strip().lower()
    if unit_norm in {"", "nm", "nanometer", "nanometers"}:
        pass
    elif unit_norm in {"micron", "microns", "micrometer", "micrometers", "um", "μm"}:
        arr = arr * 1000.0
    elif unit_norm in {"angstrom", "angstroms", "å", "a"}:
        arr = arr * 0.1
    else:
        if np.nanmax(arr) < 100:
            arr = arr * 1000.0
        else:
            raise ValueError(f"Unsupported wavelength unit: {unit}")
    if arr.ndim != 1:
        raise ValueError("Wavelength data must be one-dimensional")
    if np.any(np.diff(arr) <= 0):
        raise ValueError("Wavelength grid must be strictly increasing")
    return arr.astype(np.float64)


def _normalize_reflectance(values: np.ndarray, unit: str | None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    unit_norm = (unit or "").strip().lower()
    if unit_norm in {"", "fraction", "unitless", "reflectance"}:
        pass
    elif unit_norm in {"percent", "%", "percentage"}:
        arr = arr / 100.0
    else:
        if np.nanmax(arr) > 1.2:
            arr = arr / 100.0
        else:
            raise ValueError(f"Unsupported reflectance unit: {unit}")
    if np.any(arr < -1e-8) or np.any(arr > 1 + 1e-8):
        raise ValueError("Reflectance values must be within [0, 1]")
    arr = np.clip(arr, 0.0, 1.0)
    return arr.astype(np.float64)


def _infer_canonical_name(metadata: Mapping[str, str], file: Path) -> str:
    for key in _METADATA_KEYS:
        value = metadata.get(key)
        if value:
            return _clean_name(value)
    return _clean_name(file.stem)


def _collect_aliases(metadata: Mapping[str, str], file: Path) -> Iterator[str]:
    for key in _ALIAS_KEYS:
        value = metadata.get(key)
        if value:
            for part in re.split(r"[;,]", value):
                clean = _clean_name(part)
                if clean:
                    yield clean
    yield _clean_name(file.stem)
    parent = file.parent.name
    if parent and parent != str(file.stem):
        yield _clean_name(parent)


def _clean_name(name: str) -> str:
    return " ".join(name.strip().split())


def _normalize_key(name: str) -> str:
    return " ".join(name.casefold().split())


def _hash_source(path: Path) -> str:
    hasher = hashlib.sha256()
    if path.is_file():
        hasher.update(path.read_bytes())
    else:
        for file in sorted(
            p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in _DATA_EXTENSIONS
        ):
            hasher.update(str(file.relative_to(path)).encode("utf-8"))
            hasher.update(file.read_bytes())
    return hasher.hexdigest()


def _cache_path(path: Path) -> Path:
    if path.is_file():
        return path.with_suffix(path.suffix + _CACHE_SUFFIX)
    return path / _CACHE_SUFFIX


def _write_cache(cache_path: Path, src_hash: str, catalog: SPLIBCatalog) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        hash=np.array([src_hash]),
        catalog=np.array([dict(catalog)], dtype=object),
        aliases=np.array([catalog.aliases], dtype=object),
        alias_map=np.array([catalog.alias_map], dtype=object),
    )


def _load_cache(cache_path: Path, expected_hash: str) -> SPLIBCatalog | None:
    with np.load(cache_path, allow_pickle=True) as data:
        stored_hash = str(data["hash"][0])
        if stored_hash != expected_hash:
            return None
        catalog_dict = data["catalog"].item()
        aliases = data["aliases"].item()
        alias_map = data["alias_map"].item()
    return SPLIBCatalog(catalog_dict, alias_map=alias_map, aliases=aliases)

