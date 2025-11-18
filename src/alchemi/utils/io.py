from __future__ import annotations

from pathlib import Path
from typing import Protocol


class _SupportsPath(Protocol):
    """Protocol for path-like objects accepted by Path."""

    def __fspath__(self) -> str:  # pragma: no cover - runtime protocol hook
        ...


StrPath = str | Path | _SupportsPath


def read_text(path: StrPath) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_text(path: StrPath, data: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(data, encoding="utf-8")
