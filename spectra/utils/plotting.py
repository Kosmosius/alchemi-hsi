from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
import matplotlib.pyplot as plt

# Use a non-interactive backend so plotting works in headless CI.
matplotlib.use("Agg")


def plot_metric_bars(
    results: Sequence[object],
    metric: str,
    ylabel: str,
    output_path: Path,
    title: str | None = None,
    highlight_ids: Iterable[str] | None = None,
) -> None:
    """Render a bar plot for a metric across ablation runs.

    Args:
        results: Sequence of result-like objects with ``run_id`` and ``metric`` attributes.
        metric: Name of the attribute to plot on the y-axis.
        ylabel: Label for the y-axis.
        output_path: Where to write the PNG.
        title: Optional plot title.
        highlight_ids: Optional iterable of run_ids to highlight.
    """
    run_ids: list[str] = []
    values: list[float] = []
    for res in results:
        run_ids.append(getattr(res, "run_id"))
        values.append(float(getattr(res, metric)))

    if not run_ids:
        return

    highlight_set = set(highlight_ids or [])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(min(12, len(run_ids) * 0.75 + 3), 4.0))

    colors = [
        "#f58518" if rid in highlight_set else "#4c78a8"
        for rid in run_ids
    ]

    xs = range(len(run_ids))
    bars = ax.bar(xs, values, color=colors)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("config")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(run_ids, rotation=70, ha="right", fontsize=8)
    if title:
        ax.set_title(title)

    for bar, value in zip(bars, values, strict=True):
        ax.annotate(
            f"{value:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
