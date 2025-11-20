from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Baseline:
    name: str
    no_spatial_mask: bool = False
    no_posenc: bool = False

    def as_dict(self, steps: int) -> dict[str, str | int | float]:
        return {
            "name": self.name,
            "no_spatial_mask": self.no_spatial_mask,
            "no_posenc": self.no_posenc,
            "steps": steps,
            "score": simulate_mae(
                steps, no_spatial_mask=self.no_spatial_mask, no_posenc=self.no_posenc
            ),
        }


BASELINES: tuple[Baseline, ...] = (
    Baseline("full"),
    Baseline("spectral-only", no_spatial_mask=True),
    Baseline("no-posenc", no_posenc=True),
)


def available_baselines() -> tuple[Baseline, ...]:
    return BASELINES


def simulate_mae(steps: int, *, no_spatial_mask: bool = False, no_posenc: bool = False) -> float:
    base_score = float(max(steps, 1))
    if no_spatial_mask:
        base_score -= 0.2 * steps
    if no_posenc:
        base_score -= 0.1 * steps
    return base_score


def _write_ablation_csv(path: Path, steps: int, baselines: Iterable[Baseline]) -> None:
    rows = [b.as_dict(steps) for b in baselines]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MAE baselines and ablations")
    parser.add_argument("--steps", type=int, default=100, help="Training steps to simulate")
    parser.add_argument(
        "--baseline",
        choices=[b.name for b in BASELINES],
        default=BASELINES[0].name,
        help="Baseline configuration to run",
    )
    parser.add_argument(
        "--list-baselines",
        action="store_true",
        help="List available MAE baselines",
    )
    parser.add_argument(
        "--write-ablation",
        type=Path,
        help="Write ablation CSV covering all baselines",
    )
    return parser.parse_args(argv)


def _run_selected(baseline: Baseline, steps: int) -> float:
    score = simulate_mae(steps, no_spatial_mask=baseline.no_spatial_mask, no_posenc=baseline.no_posenc)
    print(f"{baseline.name} score={score:.3f}")
    return score


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.list_baselines:
        for b in BASELINES:
            print(b.name)
        return 0

    if args.write_ablation:
        _write_ablation_csv(args.write_ablation, args.steps, BASELINES)
        return 0

    baseline = next(b for b in BASELINES if b.name == args.baseline)
    _run_selected(baseline, args.steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
