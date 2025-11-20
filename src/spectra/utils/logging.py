from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional


@dataclass
class CSVLogger:
    path: Path
    fieldnames: Iterable[str] | None = None
    _writer: Optional[csv.DictWriter] = field(init=False, default=None)
    _file: Optional[object] = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", newline="")
        if self.fieldnames is not None:
            self._writer = csv.DictWriter(self._file, fieldnames=list(self.fieldnames))
            self._writer.writeheader()

    def log(self, row: Dict) -> None:
        if self._writer is None:
            self._writer = csv.DictWriter(self._file, fieldnames=sorted(row))  # type: ignore[arg-type]
            self._writer.writeheader()
        assert self._writer is not None and self._file is not None
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:  # pragma: no cover - trivial
        if self._file is not None:
            self._file.close()


def maybe_init_wandb(config: Dict) -> Optional[object]:
    if os.environ.get("WANDB_API_KEY") is None:
        return None
    try:  # pragma: no cover - optional dependency
        import wandb

        return wandb.init(project=config.get("project", "spectra"), config=config)
    except Exception:
        return None


__all__ = ["CSVLogger", "maybe_init_wandb"]
