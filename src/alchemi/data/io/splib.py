"""Loader for the USGS Spectral Library (SPLIB)."""

# mypy: ignore-errors

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
    canonical: str
    alias_tokens: Sequence[str]
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
        *args,
        alias_map: Mapping[str, str] | None = None,
        aliases: Mapping[str, Sequence[str]] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.alias_map = dict(alias_map or {})
        self.aliases = {k: sorted(v) for k, v in (aliases or {}).items()}

    def canonical_name(self, name: str) -> str:
        """Return canonical name or a normalized fallback key."""
        key = _normalize_key(name)
        return self.alias_map.get(key, key)

    def resolve(self, name: str) -> list[Spectrum]:
        """Resolve *name* or alias to a list of spectra."""
        canonical = self.canonical_name(name)
        if canonical not in self:
            raise KeyError(name)
        return self[canonical]

    def add_spectrum(self, name: str, spectrum: Spectrum) -> None:
        """Add a spectrum under a (possibly new) material name."""
        key = _normalize_key(name)
        canonical = self.alias_map.get(key, _clean_name(name))
        self.setdefault(canonical, []).append(spectrum)
        # Keep alias structures roughly consistent
        self.alias_map.setdefault(key, canonical)
        self.aliases.setdefault(canonical, [])
        if name not in self.aliases[canonical]:
            self.aliases[canonical].append(name)
            self.aliases[canonical].sort()

    def register_alias(self, alias: str, canonical: str) -> None:
        """Register an additional alias for an existing canonical name."""
        canonical_clean = _clean_name(canonical)
        alias_clean = _clean_name(alias)
        alias_key = _normalize_key(alias_clean)
        self.alias_map[alias_key] = canonical_clean
        self.aliases.setdefault(canonical_clean, [])
        if alias_clean not in self.aliases[canonical_clean]:
            self.aliases[canonical_clean].append(alias_clean)
            self.aliases[canonical_clean].sort()


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


def load_splib(
    src: str | os.PathLike[str] | Iterable[str | os.PathLike[str]],
    *,
    use_cache: bool = True,
) -> SPLIBCatalog:
    """Parse SPLIB spectra from *src* into a :class:`SPLIBCatalog`.

    ``src`` may be:

    * a single file (ASCII/CSV-style table),
    * a directory containing multiple spectra files, or
    * an iterable of such paths.

    When a single path is provided, a small cache is used (if ``use_cache`` is
    True). When multiple paths are provided, all spectra are loaded and merged
    without caching.
    """

    if isinstance(src, (str, os.PathLike)):
        path = Path(src)
        return _load_splib_from_path(path, use_cache=use_cache)

    # Iterable of paths: aggregate entries from all, no shared cache
    paths = [Path(p) for p in src]
    if not paths:
        raise ValueError("No SPLIB sources provided")

    entries: list[_RawEntry] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"SPLIB source not found: {path}")
        entries.extend(_parse_entries(path))

    if not entries:
        raise ValueError(f"No SPLIB spectra found in {paths}")

    return _build_catalog(entries)


def _load_splib_from_path(path: Path, *, use_cache: bool) -> SPLIBCatalog:
    if not path.exists():
        raise FileNotFoundError(f"SPLIB source not found: {path}")

    src_hash = _hash_source(path)
    cache_path = _cache_path(path)

    if use_cache and cache_path.exists():
        try:
            catalog = _load_cache(cache_path, src_hash)
        except Exception:
            catalog = None
        if catalog is not None:
            return catalog

    entries = list(_parse_entries(path))
    if not entries:
        raise ValueError(f"No SPLIB spectra found in {path}")

    catalog = _build_catalog(entries)

    if use_cache:
        _write_cache(cache_path, src_hash, catalog)

    return catalog


def _build_catalog(entries: Sequence[_RawEntry]) -> SPLIBCatalog:
    alias_lookup: dict[str, str] = {}
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

    aliases_final: dict[str, list[str]] = {k: sorted(v) for k, v in aliases_by_canonical.items()}

    catalog = SPLIBCatalog(alias_map=alias_lookup, aliases=aliases_final)
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
                "aliases": aliases_final.get(entry.canonical, []),
                "source": str(entry.source),
            },
        )
        catalog.setdefault(entry.canonical, []).append(spectrum)

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
    metadata: dict[str, str] = {}
    data_lines: list[str] = []
    in_data = False

    for line in content:
        stripped = line.strip()
        if not stripped:
            if in_data:
                data_lines.append(stripped)
            continue

        # Comment-style metadata lines
        if not in_data and stripped.startswith("#"):
            key, value = _parse_metadata_line(stripped[1:].strip())
            if key:
                metadata[key] = value
            continue

        tokens = _split_columns(stripped)
        if not tokens:
            continue

        # Header row: try to infer metadata such as units
        if not in_data and not _tokens_are_numeric(tokens):
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
    reflectance_norm = _normalize_reflectance(reflectance, ref_unit)

    canonical = _infer_canonical_name(metadata, file)
    alias_tokens = list(_collect_aliases(metadata, file))

    yield _RawEntry(
        canonical=canonical,
        alias_tokens=alias_tokens,
        wavelengths_nm=wavelengths_nm,
        reflectance=reflectance_norm,
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


def _split_columns(line: str) -> list[str]:
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
    data: list[list[float]] = []
    for line in lines:
        if not line:
            continue
        tokens = _split_columns(line)
        if len(tokens) < 2:
            continue
        try:
            row = [float(tokens[0]), float(tokens[1])]
        except ValueError as exc:
            msg = f"Non-numeric data row in SPLIB file: {line!r}"
            raise ValueError(msg) from exc
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
        # Heuristic: if values look like microns, convert; otherwise bail out.
        if np.nanmax(arr) < 100:
            arr = arr * 1000.0
        else:
            msg = f"Unsupported wavelength unit: {unit}"
            raise ValueError(msg)
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
        # Heuristic: if values are >1, assume percent
        if np.nanmax(arr) > 1.2:
            arr = arr / 100.0
        else:
            msg = f"Unsupported reflectance unit: {unit}"
            raise ValueError(msg)
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
        if not value:
            continue
        for part in re.split(r"[;,]", value):
            clean = _clean_name(part)
            if clean:
                yield clean
    # Always include stem and parent directory as fall-back aliases
    stem = _clean_name(file.stem)
    if stem:
        yield stem
    parent = file.parent.name
    if parent and parent != file.stem:
        parent_clean = _clean_name(parent)
        if parent_clean:
            yield parent_clean


def _clean_name(name: str) -> str:
    return " ".join(name.strip().split())


def _normalize_key(name: str) -> str:
    # Case-insensitive, whitespace-normalised key
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
