"""Plotting helpers for small synthetic experiments."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Protocol

import matplotlib.pyplot as plt


class _HasRunId(Protocol):
    run_id: str


def plot_metric_bars(
    results: Sequence[_HasRunId],
    *,
    metric: str,
    ylabel: str,
    output_path: str | Path,
    title: str,
    highlight_ids: Iterable[str] | None = None,
) -> None:
    """Create and save a simple bar plot for the provided metric.

    The ``results`` sequence is expected to contain objects exposing ``run_id`` and
    the requested ``metric`` attribute. When no results are provided the function
    returns without side effects.
    """

    if not results:
        return

    highlight = set(highlight_ids or [])
    run_ids = [r.run_id for r in results]
    values = [float(getattr(r, metric)) for r in results]
    labels = [f"{idx + 1}" for idx in range(len(results))]

    colors = ["tab:blue" if rid not in highlight else "tab:orange" for rid in run_ids]
    edges = [2.5 if rid in highlight else 0.5 for rid in run_ids]

    fig, ax = plt.subplots(figsize=(max(6, len(run_ids) * 0.45), 4))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=edges)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("config index")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=0, labelsize=8)

    # Annotate highlighted bars with their run_id to ease discovery.
    for bar, rid in zip(bars, run_ids):
        if rid in highlight:
            height = bar.get_height()
            ax.annotate(
                rid,
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=45,
            )

    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
