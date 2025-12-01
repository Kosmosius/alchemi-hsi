"""Simple dataset catalog loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping

import yaml

DEFAULT_CATALOG_PATHS: list[Path] = [Path("resources/examples/catalog.yaml"), Path("resources/examples/catalog.json")]


class SceneCatalog:
    """Registry of scene paths grouped by task/sensor/split."""

    def __init__(self, sources: list[Path] | None = None) -> None:
        self.sources = sources or DEFAULT_CATALOG_PATHS
        self._index: Dict[str, Dict[str, Dict[str, list[Path]]]] = {}
        self._load_sources()

    def _load_sources(self) -> None:
        for path in self.sources:
            if not path.exists():
                continue
            data = _load_mapping(path)
            for task, sensors in data.items():
                task_bucket = self._index.setdefault(task, {})
                for sensor_id, splits in sensors.items():
                    sensor_bucket = task_bucket.setdefault(sensor_id, {})
                    for split, entries in splits.items():
                        sensor_bucket.setdefault(split, []).extend(Path(e) for e in entries)

    def get_scenes(self, split: str, sensor_id: str, task: str) -> List[Path]:
        """Return the list of scenes matching the provided filters."""

        return list(self._index.get(task, {}).get(sensor_id, {}).get(split, []))

    def available_tasks(self) -> list[str]:
        return list(self._index.keys())


def _load_mapping(path: Path) -> Mapping[str, Mapping[str, Mapping[str, list[str]]]]:
    if path.suffix in {".yaml", ".yml"}:
        return yaml.safe_load(path.read_text())
    if path.suffix == ".json":
        return json.loads(path.read_text())
    raise ValueError(f"Unsupported catalog format: {path}")
