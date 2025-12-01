"""Preprocessing CLI for lab index building and dataset preparation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Sequence

from alchemi.config import load_experiment_config


@dataclass
class Preprocessor:
    """Dispatcher for preprocessing utilities.

    The heavy lifting for each task should live alongside the data/ingest
    modules; the CLI simply routes based on flags so that data preparation can
    be scripted uniformly.
    """

    experiment: str
    build_lab_index: bool
    build_datasets: bool
    config_root: str | None = None

    def run(self) -> None:
        cfg = load_experiment_config(self.experiment, config_root=self.config_root)
        print(f"[preprocess] Loaded experiment: {cfg.experiment_name}")
        if self.build_lab_index:
            self._build_lab_index(cfg)
        if self.build_datasets:
            self._build_dataset_indices(cfg)

    def _build_lab_index(self, cfg) -> None:
        print("[preprocess] Building lab embedding index (placeholder)")
        # TODO: connect to the lab embedding export and FAISS/torch indexers.

    def _build_dataset_indices(self, cfg) -> None:
        print("[preprocess] Precomputing dataset indices / tiles (placeholder)")
        # TODO: hook into dataset tiling/tokenisation utilities when available.


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Precompute lab embeddings or dataset indices for ALCHEMI",
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
        "--build-lab-index",
        action="store_true",
        help="Generate or refresh the lab embedding index",
    )
    parser.add_argument(
        "--build-datasets",
        action="store_true",
        help="Precompute dataset indices or tiled chips",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    preprocessor = Preprocessor(
        experiment=args.experiment,
        build_lab_index=args.build_lab_index,
        build_datasets=args.build_datasets,
        config_root=args.config_root,
    )
    preprocessor.run()


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
