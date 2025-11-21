"""Plotting helpers for small synthetic experiments."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Protocol


class _HasRunId(Protocol):
    run_id: str


def plot_metric_bars(
    results: Sequence[_HasRunId],
    *,
    metric: str,
    ylabel: str,
    output_path: str | Path,
    title: str | None = None,
    highlight_ids: Iterable[str] | None = None,
) -> None:
    """Save a simple bar plot for a metric extracted from a results sequence.

    Each result is expected to expose the metric attribute and a ``run_id`` string
    field. The function is tolerant of missing matplotlib installations: if the
    backend is unavailable, the function exits without raising to keep CLI flows
    lightweight.
    """

    try:
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - optional dependency
        return

    highlight = set(highlight_ids or [])
    run_ids = [r.run_id for r in results]
    values = [float(getattr(r, metric)) for r in results]
    colors = ["tab:blue" if rid not in highlight else "tab:orange" for rid in run_ids]

    fig, ax = plt.subplots(figsize=(max(6, len(run_ids) * 0.4), 4))
    ax.bar(run_ids, values, color=colors)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("run_id")
    ax.set_title(title or metric)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
