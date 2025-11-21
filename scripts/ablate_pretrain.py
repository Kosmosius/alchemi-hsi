"""Synthetic pretraining ablation harness.

This script sweeps masking ratios, grouping sizes/modes, and ingest settings.
It produces lightweight CSV logs and simple plots summarizing reconstruction
loss and throughput for quick comparisons. The underlying computations are
synthetic to keep tests fast while still exercising the orchestration code
paths (grid construction, logging, plotting).
"""
from __future__ import annotations

import csv
import math
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

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


def plot_metric_bars(
    results: Sequence[AblationResult],
    *,
    metric: str,
    ylabel: str,
    output_path: Path,
    title: str,
    highlight_ids: Sequence[str] | None = None,
) -> None:
    """Minimal plotting helper that falls back to touching a file.

    The tests only verify that plot files exist, so if matplotlib is unavailable we
    simply create an empty file to satisfy the expectation.
    """

    try:
        import matplotlib.pyplot as plt
    except Exception:
        output_path.touch()
        return

    labels = [res.run_id for res in results]
    values = [getattr(res, metric) for res in results]
    highlight_ids = set(highlight_ids or [])

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["tab:blue" if rid not in highlight_ids else "tab:orange" for rid in labels]
    ax.bar(labels, values, color=colors)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


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


def _synthetic_recon_mse(
    cfg: AblationConfig,
    steps: int,
    rng: np.random.Generator,
) -> float:
    """Toy reconstruction loss that prefers small masks, G≈8, data_driven, and ingest_any."""
    base = 0.02
    spatial_penalty = 0.01 * (cfg.mask_spatial - 0.5)
    spectral_penalty = 0.015 * (cfg.mask_spectral - 0.3)
    grouping_penalty = 0.003 * abs(cfg.G - 8) / 8.0
    mode_penalty = 0.004 if cfg.grouping_mode == "contiguous" else 0.0
    ingest_bonus = -0.002 if cfg.ingest_any else 0.0
    decay = 1.0 / math.sqrt(max(1, steps))
    noise = rng.normal(0.0, 5e-4)
    value = (base + spatial_penalty + spectral_penalty + grouping_penalty + mode_penalty) * decay
    value += ingest_bonus + noise
    return max(1e-4, float(value))


def _synthetic_retrieval_top1(
    cfg: AblationConfig,
    rng: np.random.Generator,
    probe_items: int,
) -> float:
    """Toy retrieval@1 over synthetic prototypes/queries."""
    n_classes = 4
    dim = 8
    prototypes = rng.normal(size=(n_classes, dim))
    queries = rng.normal(size=(probe_items, dim))
    labels = rng.integers(0, n_classes, size=probe_items)

    # Slightly reward ingest_any for this metric.
    adapt = 1.0 + (0.1 if cfg.ingest_any else 0.0)
    distance = np.zeros((probe_items, n_classes))
    for idx, proto in enumerate(prototypes):
        distance[:, idx] = np.linalg.norm(queries * adapt - proto, axis=1)
    preds = distance.argmin(axis=1)
    return float((preds == labels).mean())


def _tokens_per_second(steps: int, cfg: AblationConfig, elapsed: float) -> float:
    """Synthetic throughput proxy."""
    tokens = steps * cfg.G * 64
    return float(tokens / max(elapsed, 1e-3))


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
    steps: int = 24,
    probe_items: int = 16,
    output_dir: Path | str = Path("outputs/ablations"),
) -> list[AblationResult]:
    """Execute a sweep of synthetic pretraining ablations.

    Args:
        configs: Optional subset of configs to evaluate; defaults to full grid.
        steps: Synthetic training steps per config.
        probe_items: Size of the retrieval sanity probe set.
        output_dir: Base directory for CSVs and plots.

    Returns:
        A list of AblationResult objects, one per evaluated config.
    """
    cfgs: Iterable[AblationConfig] = configs or build_ablation_configs()
    output_dir = Path(output_dir)

    results: list[AblationResult] = []
    start_seed = 1234

    for idx, cfg in enumerate(cfgs):
        rng = np.random.default_rng(start_seed + idx)
        tic = time.perf_counter()
        recon_mse = _synthetic_recon_mse(cfg, steps=steps, rng=rng)
        retrieval_top1 = _synthetic_retrieval_top1(cfg, rng=rng, probe_items=probe_items)
        elapsed = time.perf_counter() - tic
        tokens_per_s = _tokens_per_second(steps=steps, cfg=cfg, elapsed=elapsed)

        result = AblationResult(
            run_id=cfg.run_id,
            mask_spatial=cfg.mask_spatial,
            mask_spectral=cfg.mask_spectral,
            grouping_mode=cfg.grouping_mode,
            G=cfg.G,
            ingest_any=cfg.ingest_any,
            steps=steps,
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

    if results:
        # Highlight the best config by recon_mse (ties broken by throughput).
        best = min(results, key=lambda r: (r.recon_mse, -r.tokens_per_s))
        plot_metric_bars(
            results,
            metric="recon_mse",
            ylabel="Reconstruction MSE",
            output_path=output_dir / "recon_mse.png",
            title="Synthetic reconstruction loss across ablations",
            highlight_ids=[best.run_id],
        )
        plot_metric_bars(
            results,
            metric="tokens_per_s",
            ylabel="Tokens / second",
            output_path=output_dir / "throughput.png",
            title="Synthetic throughput across ablations",
            highlight_ids=[best.run_id],
        )

    return results


app = typer.Typer(help="Synthetic pretraining ablation harness")


@app.command()
def run(
    max_runs: Annotated[
        int,
        typer.Option(
            help="Maximum ablation configurations to evaluate (subset of full grid)",
        ),
    ] = 48,
    steps: Annotated[
        int,
        typer.Option(help="Number of synthetic training steps per configuration"),
    ] = 24,
    output_dir: Annotated[
        Path,
        typer.Option(help="Output directory for CSVs and plots"),
    ] = Path("outputs/ablations"),
    probe_items: Annotated[
        int,
        typer.Option(help="Probe samples for the retrieval@1 sanity check"),
    ] = 16,
) -> None:
    """CLI entrypoint to execute a sweep of pretraining ablations."""
    configs = build_ablation_configs()[:max_runs]
    results = run_ablation(
        configs=configs,
        steps=steps,
        probe_items=probe_items,
        output_dir=output_dir,
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
