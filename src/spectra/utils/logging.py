from __future__ import annotations

import csv
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO


@dataclass
class CSVLogger:
    path: Path
    fieldnames: Iterable[str] | None = None
    _writer: csv.DictWriter[str] | None = field(init=False, default=None)
    _file: TextIO | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", newline="")
        if self.fieldnames is not None:
            self._writer = csv.DictWriter(self._file, fieldnames=list(self.fieldnames))
            self._writer.writeheader()

    def log(self, row: dict[str, Any]) -> None:
        if self._writer is None:
            assert self._file is not None
            self._writer = csv.DictWriter(self._file, fieldnames=sorted(row))
            self._writer.writeheader()
        assert self._writer is not None and self._file is not None
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:  # pragma: no cover - trivial
        if self._file is not None:
            self._file.close()


def maybe_init_wandb(config: dict[str, Any]) -> object | None:
    if os.environ.get("WANDB_API_KEY") is None:
        return None
    try:  # pragma: no cover - optional dependency
        import wandb

        run: object = wandb.init(
            project=str(config.get("project", "spectra")),
            config=config,
        )
        return run
    except Exception:
        return None


__all__ = ["CSVLogger", "maybe_init_wandb"]
