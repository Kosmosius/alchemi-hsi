"""Pretraining ablation harness driven by the MAE trainer."""

from __future__ import annotations

import csv
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer

from alchemi.training.config import TrainCfg
from alchemi.training.trainer import run_pretrain_mae
from spectra.utils.plotting import plot_metric_bars

# Search space
MASK_SPATIAL = (0.5, 0.75)
MASK_SPECTRAL = (0.3, 0.5)
GROUPING_MODES = ("contiguous", "data_driven")
GROUP_SIZES = (4, 8, 16)
INGEST_ANY = (False, True)

OUTPUT_COLUMNS = [
    "run_id",
    "mask_spatial",
    "mask_spectral",
    "grouping_mode",
    "G",
    "ingest_any",
    "steps",
    "recon_mse",
    "tokens_per_s",
    "retrieval_top1",
]


@dataclass(frozen=True)
class AblationConfig:
    mask_spatial: float
    mask_spectral: float
    grouping_mode: str
    G: int
    ingest_any: bool

    @property
    def run_id(self) -> str:
        """Deterministic, human-readable identifier encoding the config."""
        ingest_tag = "any" if self.ingest_any else "spectralearth"
        return (
            f"ms{self.mask_spatial:.2f}_mspec{self.mask_spectral:.2f}_"
            f"mode-{self.grouping_mode}_G{self.G}_{ingest_tag}"
        )


@dataclass
class AblationResult:
    run_id: str
    mask_spatial: float
    mask_spectral: float
    grouping_mode: str
    G: int
    ingest_any: bool
    steps: int
    recon_mse: float
    tokens_per_s: float
    retrieval_top1: float

    def to_row(self) -> dict[str, str | float | int]:
        return {field: getattr(self, field) for field in OUTPUT_COLUMNS}


def build_ablation_configs() -> list[AblationConfig]:
    """Construct the full ablation grid."""
    configs: list[AblationConfig] = []
    for ms in MASK_SPATIAL:
        for mt in MASK_SPECTRAL:
            for mode in GROUPING_MODES:
                for g in GROUP_SIZES:
                    for ingest in INGEST_ANY:
                        configs.append(
                            AblationConfig(
                                mask_spatial=ms,
                                mask_spectral=mt,
                                grouping_mode=mode,
                                G=g,
                                ingest_any=ingest,
                            )
                        )
    return configs


def _ablation_to_config(
    cfg: AblationConfig, *, steps: int, output_dir: Path, seed: int
) -> dict[str, Any]:
    """Convert an ablation config into a MAE TrainCfg payload."""

    mask_path = output_dir / cfg.run_id / "mask.pt"
    mask_path.parent.mkdir(parents=True, exist_ok=True)

    train_cfg = TrainCfg(
        batch_size=2,
        lr=1e-3,
        max_steps=steps,
        log_every=max(1, steps // 2),
        embed_dim=max(4, cfg.G),
        n_heads=2,
        depth=2 if cfg.grouping_mode == "contiguous" else 3,
        basis_K=max(4, cfg.G),
        spatial_mask_ratio=cfg.mask_spatial,
        spectral_mask_ratio=cfg.mask_spectral,
        no_posenc=cfg.grouping_mode == "contiguous",
        no_spatial_mask=not cfg.ingest_any,
        mask_path=str(mask_path),
        deterministic=True,
        seed=seed,
    )

    return {
        "global": {"device": "cpu", "dtype": "float32", "deterministic": True, "seed": seed},
        "train": train_cfg.model_dump(),
    }


def _write_csv(path: Path, result: AblationResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(result.to_row())


def run_ablation(
    configs: Sequence[AblationConfig] | None = None,
    *,
    steps_per_config: int = 10,
    output_dir: Path | str = Path("outputs/ablations"),
    plot: bool = True,
) -> list[AblationResult]:
    """Execute a sweep of MAE pretraining ablations.

    Args:
        configs: Optional subset of configs to evaluate; defaults to full grid.
        steps_per_config: Number of MAE training steps to run per configuration.
        output_dir: Base directory for CSVs and plots.
        plot: Whether to generate summary plots for the sweep.

    Returns:
        A list of AblationResult objects, one per evaluated config.
    """
    cfgs: Iterable[AblationConfig] = configs or build_ablation_configs()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[AblationResult] = []
    start_seed = 1234
    cwd = Path.cwd()

    for idx, cfg in enumerate(cfgs):
        seed = start_seed + idx
        mae_config = _ablation_to_config(
            cfg, steps=steps_per_config, output_dir=output_dir, seed=seed
        )

        # Keep trainer artifacts scoped to the output directory for cleanliness.
        os.chdir(output_dir)
        try:
            stats, recon_mse = run_pretrain_mae(mae_config, seed_override=seed, return_loss=True)
        finally:
            os.chdir(cwd)

        tokens_per_s = float(stats.tokens_per_s)
        retrieval_top1 = 0.0  # Placeholder until retrieval metrics are exposed.

        result = AblationResult(
            run_id=cfg.run_id,
            mask_spatial=cfg.mask_spatial,
            mask_spectral=cfg.mask_spectral,
            grouping_mode=cfg.grouping_mode,
            G=cfg.G,
            ingest_any=cfg.ingest_any,
            steps=steps_per_config,
            recon_mse=recon_mse,
            tokens_per_s=tokens_per_s,
            retrieval_top1=retrieval_top1,
        )
        results.append(result)

        csv_path = output_dir / f"{cfg.run_id}.csv"
        _write_csv(csv_path, result)

        typer.echo(
            f"[run {cfg.run_id}] "
            f"recon_mse={recon_mse:.5f} "
            f"retrieval_top1={retrieval_top1:.3f} "
            f"tokens_per_s={tokens_per_s:.1f}"
        )

    if results and plot:
        # Highlight the best config by recon_mse (ties broken by throughput).
        best = min(results, key=lambda r: (r.recon_mse, -r.tokens_per_s))
        plot_metric_bars(
            results,
            metric="recon_mse",
            ylabel="Reconstruction MSE",
            output_path=output_dir / "recon_mse.png",
            title="MAE reconstruction loss across ablations",
            highlight_ids=[best.run_id],
        )
        plot_metric_bars(
            results,
            metric="tokens_per_s",
            ylabel="Tokens / second",
            output_path=output_dir / "throughput.png",
            title="MAE throughput across ablations",
            highlight_ids=[best.run_id],
        )

    return results


app = typer.Typer(help="MAE pretraining ablation harness")


@app.command()
def run(
    max_runs: int = typer.Option(
        48,
        help="Maximum ablation configurations to evaluate (subset of full grid)",
    ),
    steps_per_config: int = typer.Option(
        24,
        "--steps-per-config",
        "--steps",
        help="Number of MAE training steps per configuration",
    ),
    output_dir: Path = typer.Option(
        Path("outputs/ablations"),
        help="Output directory for CSVs and plots",
    ),
    plot: bool = typer.Option(
        True,
        help="Generate matplotlib summary plots (disable in headless CI with --no-plot)",
    ),
) -> None:
    """CLI entrypoint to execute a sweep of pretraining ablations."""
    configs = build_ablation_configs()[:max_runs]
    results = run_ablation(
        configs=configs,
        steps_per_config=steps_per_config,
        output_dir=output_dir,
        plot=plot,
    )
    if results:
        best = min(results, key=lambda r: (r.recon_mse, -r.tokens_per_s))
        typer.echo(
            "Lowest MSE configuration: "
            f"mask_spatial={best.mask_spatial}, "
            f"mask_spectral={best.mask_spectral}, "
            f"mode={best.grouping_mode}, "
            f"G={best.G}, "
            f"ingest_any={best.ingest_any} "
            f"(recon_mse={best.recon_mse:.5f}, tokens_per_s={best.tokens_per_s:.1f})"
        )


if __name__ == "__main__":
    app()
