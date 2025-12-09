"""Helpers for resolving project-relative paths.

The config loader relies on these utilities to resolve relative paths from YAML
files (e.g., data roots or resource directories) against the repository root.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from alchemi.utils.io import StrPath


def find_project_root(
    start: StrPath | None = None, markers: Iterable[str] = ("pyproject.toml", ".git")
) -> Path:
    """Locate the project root by searching for a marker file.

    Parameters
    ----------
    start:
        Optional starting path. Defaults to the directory containing this file.
    markers:
        Filenames that signal the repository root.
    """

    start_path = Path(start or __file__).resolve()
    for candidate in (start_path, *start_path.parents):
        if any((candidate / marker).exists() for marker in markers):
            return candidate if candidate.is_dir() else candidate.parent
    raise FileNotFoundError("Could not locate project root; missing marker files")


def resolve_path(path: StrPath, base: StrPath | None = None) -> Path:
    """Resolve ``path`` relative to ``base`` (or the project root)."""

    raw = Path(path)
    if raw.is_absolute():
        return raw
    if base is not None:
        return Path(base).expanduser().resolve() / raw
    return find_project_root() / raw
