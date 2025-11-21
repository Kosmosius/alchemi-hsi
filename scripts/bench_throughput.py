from __future__ import annotations

"""Benchmark training throughput and record run-to-run variance."""

import argparse
import csv
import sys
import tempfile
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from statistics import mean, pstdev

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from alchemi.training.trainer import run_align, run_pretrain_mae
from alchemi.utils.logging import ThroughputStats, get_logger

_LOG = get_logger(__name__)


def _build_runner(mode: str) -> Callable[[str], ThroughputStats]:
    if mode == "mae":
        return run_pretrain_mae
    if mode == "align":
        return run_align
    raise ValueError(f"Unsupported mode: {mode}")


def _prepare_config(config_path: str, max_steps: int | None) -> tuple[str, bool]:
    """Optionally override train.max_steps in a temporary config file."""
    if max_steps is None:
        return config_path, False

    config = yaml.safe_load(Path(config_path).read_text())
    config.setdefault("train", {})["max_steps"] = max_steps
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(config_path).suffix)
    tmp.write(yaml.safe_dump(config).encode("utf-8"))
    tmp.flush()
    return tmp.name, True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark training throughput and run-to-run variance."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.mae.yaml",
        help="Path to training YAML config.",
    )
    parser.add_argument(
        "--mode",
        choices=["mae", "align"],
        default="mae",
        help="Which training loop to benchmark.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of independent runs to average over.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Override train.max_steps in the config for this benchmark.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/throughput.csv"),
        help="Where to store throughput results.",
    )
    args = parser.parse_args()

    runner = _build_runner(args.mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    config_override, needs_cleanup = _prepare_config(args.config, args.max_steps)
    rows: list[dict[str, float]] = []

    try:
        for i in range(1, args.runs + 1):
            stats = runner(config_override)
            stats_dict = asdict(stats)
            rows.append({"run": float(i), **stats_dict})
            _LOG.info(
                "[%s run %d/%d] tokens/s=%.1f gb/s=%.2f peak_mem=%.2fGB",
                args.mode,
                i,
                args.runs,
                stats.tokens_per_s,
                stats.gb_per_s or 0.0,
                stats.peak_mem_gb or 0.0,
            )
    finally:
        if needs_cleanup:
            Path(config_override).unlink(missing_ok=True)

    tokens = [row["tokens_per_s"] for row in rows]
    avg_tokens = mean(tokens) if tokens else 0.0
    variability = pstdev(tokens) / avg_tokens if avg_tokens else 0.0

    _LOG.info("Tokens/s coefficient of variation across runs: %.2f%%", variability * 100)
    if variability > 0.10:
        _LOG.warning(
            "Throughput variability exceeds 10%% (cv=%.2f%%). "
            "Consider rerunning or pinning seeds.",
            variability * 100,
        )

    fieldnames = [
        "run",
        "tokens_per_s",
        "gb_per_s",
        "peak_mem_gb",
        "step_time_s",
        "tokens",
        "num_bytes",
        "cv_tokens",
    ]
    with args.output.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = row.copy()
            out["cv_tokens"] = variability
            writer.writerow(out)

    print(
        f"Stored throughput results in {args.output} "
        f"(cv={variability * 100:.2f}% over {args.runs} runs)"
    )


if __name__ == "__main__":
    main()
