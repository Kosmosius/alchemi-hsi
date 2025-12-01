"""Training CLI stub that wires experiment configs into the runtime."""

from __future__ import annotations

import argparse
import pprint
from typing import Sequence

from alchemi.config import load_experiment_config


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="ALCHEMI training entrypoint (config only)")
    parser.add_argument(
        "experiment",
        help="Name of the experiment config (without .yaml) or a direct path",
    )
    parser.add_argument(
        "--config-root",
        dest="config_root",
        default=None,
        help="Optional override for the configs directory",
    )
    args = parser.parse_args(argv)

    cfg = load_experiment_config(args.experiment, config_root=args.config_root)
    print(f"Loaded experiment: {cfg.experiment_name}")
    pprint.pp(cfg.model_dump(mode="json"))


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()

