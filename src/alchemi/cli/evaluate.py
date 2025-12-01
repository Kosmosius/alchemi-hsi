"""Evaluation CLI entrypoint for trained ALCHEMI checkpoints."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from alchemi.config import ExperimentConfig, RuntimeConfig, load_experiment_config


@dataclass
class Evaluator:
    """Thin wrapper to fan out to evaluation suites based on configuration."""

    experiment: ExperimentConfig
    runtime: RuntimeConfig
    checkpoint: Path
    only_suites: set[str] | None = None

    def run(self) -> None:
        print(f"[eval] Experiment: {self.experiment.experiment_name}")
        print(f"[eval] Checkpoint: {self.checkpoint}")
        print(f"[eval] Device: {self.runtime.device}")
        self._run_suites()

    def _run_suites(self) -> None:
        suites = self.experiment.eval
        for name, enabled in _iter_suites(suites):
            if self.only_suites and name not in self.only_suites:
                print(f"[eval] Skipping suite '{name}' (not requested)")
                continue
            if not enabled:
                print(f"[eval] Skipping suite '{name}' (disabled)")
                continue
            print(f"[eval] Running suite '{name}' on device {self.runtime.device}")
            # TODO: integrate with concrete evaluation recipes.


def _iter_suites(cfg: object) -> Iterable[tuple[str, bool]]:
    yield from (
        ("solids", cfg.solids),
        ("gas", cfg.gas),
        ("representation", cfg.representation),
        ("srf_robustness", cfg.srf_robustness),
        ("heavy_atmosphere", cfg.heavy_atmosphere),
        ("teacher_noise", cfg.teacher_noise),
        ("coverage", cfg.coverage),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained ALCHEMI checkpoint",
    )
    parser.add_argument(
        "experiment",
        help="Experiment config name (without .yaml) or direct path to a YAML file",
    )
    parser.add_argument(
        "checkpoint",
        help="Path to the trained model checkpoint to evaluate",
    )
    parser.add_argument(
        "--config-root",
        dest="config_root",
        default=None,
        help="Optional override for the configs directory",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to evaluate on (e.g. cuda:0 or cpu)",
    )
    parser.add_argument(
        "--only",
        dest="only_suites",
        choices=[
            "solids",
            "gas",
            "representation",
            "srf_robustness",
            "heavy_atmosphere",
            "teacher_noise",
            "coverage",
        ],
        action="append",
        help="Restrict evaluation to a subset of suites (repeatable)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    cfg = load_experiment_config(args.experiment, config_root=args.config_root)
    runtime = RuntimeConfig.from_mapping({"device": args.device} if args.device else None)
    checkpoint = Path(args.checkpoint)

    only = set(args.only_suites) if args.only_suites else None
    evaluator = Evaluator(cfg, runtime, checkpoint, only)
    evaluator.run()


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
