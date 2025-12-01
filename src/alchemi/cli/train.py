"""Training CLI entrypoint for running configured ALCHEMI experiments."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Iterable, Sequence

from alchemi.config import ExperimentConfig, RuntimeConfig, load_experiment_config


@dataclass
class Trainer:
    """Lightweight trainer façade that sequences the configured stages.

    This is intentionally thin: the heavy lifting is delegated to the training
    stacks under :mod:`alchemi.train` and :mod:`alchemi.training`. The class
    merely provides a structured place to plug those stacks in while keeping
    the CLI usable for configuration validation and bookkeeping.
    """

    experiment: ExperimentConfig
    runtime: RuntimeConfig

    def run(self) -> None:
        print(f"[trainer] Experiment: {self.experiment.experiment_name}")
        print(f"[trainer] Device: {self.runtime.device} (dtype={self.runtime.dtype})")
        print(f"[trainer] Seed: {self.runtime.seed} deterministic={self.runtime.deterministic}")
        self._run_stages()

    def _run_stages(self) -> None:
        stages = self.experiment.training.stages
        for name, setting in _iter_stages(stages):
            if not setting.enabled:
                print(f"[trainer] Skipping stage '{name}' (disabled)")
                continue
            print(
                "[trainer] Running stage '%s' for %s epochs (lr=%s, weight=%.2f)" % (
                    name,
                    setting.epochs if setting.epochs is not None else "configured",
                    setting.learning_rate if setting.learning_rate is not None else "default",
                    setting.loss_weight,
                )
            )
            # TODO: Wire into the actual trainer implementations when they are
            #       available for each stage.


def _iter_stages(stages: object) -> Iterable[tuple[str, object]]:
    yield from (
        ("mae", stages.mae),
        ("align", stages.align),
        ("tasks", stages.tasks),
        ("uncertainty", stages.uncertainty),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an ALCHEMI experiment (e.g. --experiment alchemi_solids_emit_v1)",
    )
    parser.add_argument(
        "experiment",
        help="Experiment config name (without .yaml) or direct path to a YAML file",
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
        help="Device to train on (e.g. cuda:0, cpu, auto)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the random seed for reproducibility",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Print the resolved experiment configuration as JSON",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    cfg = load_experiment_config(args.experiment, config_root=args.config_root)
    runtime = RuntimeConfig.from_mapping({"device": args.device} if args.device else None)
    runtime = runtime.with_seed(args.seed)

    if args.show_config:
        print(json.dumps(cfg.model_dump(mode="json"), indent=2))

    trainer = Trainer(cfg, runtime)
    trainer.run()


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()

