from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class MAEBaseline:
    model: str
    no_spatial_mask: bool = False
    no_posenc: bool = False


BASELINES: tuple[MAEBaseline, ...] = (
    MAEBaseline("mae_main", no_spatial_mask=False, no_posenc=False),
    MAEBaseline("mae_no_spatial_mask", no_spatial_mask=True, no_posenc=False),
    MAEBaseline("mae_no_posenc", no_spatial_mask=False, no_posenc=True),
)


def available_baselines() -> tuple[MAEBaseline, ...]:
    return BASELINES


def rows_from_scores(steps: int, scores: Mapping[str, float]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for baseline in BASELINES:
        score = scores[baseline.model]
        rows.append({"model": baseline.model, "steps": str(steps), "score": f"{score:.6f}"})
    return rows


def write_ablation_csv(path: str, steps: int, scores: Mapping[str, float]) -> None:
    import csv
    from pathlib import Path

    rows = rows_from_scores(steps, scores)
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("model", "steps", "score"))
        writer.writeheader()
        writer.writerows(rows)
