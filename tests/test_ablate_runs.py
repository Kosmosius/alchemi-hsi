from __future__ import annotations

import csv
import math
from pathlib import Path

from scripts.ablate_pretrain import (
    OUTPUT_COLUMNS,
    AblationConfig,
    build_ablation_configs,
    run_ablation,
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def test_pretrain_ablation_runs_full_grid_slice(tmp_path: Path) -> None:
    # Take a small slice of the grid to keep CI fast.
    configs = build_ablation_configs()[:2]
    results = run_ablation(
        configs=configs,
        steps=3,
        probe_items=6,
        output_dir=tmp_path,
        plot=False,
    )

    assert len(results) == len(configs)

    for res in results:
        csv_path = tmp_path / f"{res.run_id}.csv"
        assert csv_path.exists(), csv_path
        rows = _read_rows(csv_path)
        assert rows, "CSV should contain at least one row"
        last_row = rows[-1]
        for col in OUTPUT_COLUMNS:
            assert col in last_row, f"Missing column {col} in {csv_path}"

        assert math.isfinite(res.recon_mse)
        assert res.recon_mse > 0
        assert math.isfinite(res.tokens_per_s)
        assert res.tokens_per_s > 0
        assert 0.0 <= res.retrieval_top1 <= 1.0


def test_pretrain_ablation_runs_custom_configs(tmp_path: Path) -> None:
    # Explicit configs to sanity-check run_id and CSV naming.
    configs = [
        AblationConfig(0.5, 0.3, "contiguous", 4, False),
        AblationConfig(0.5, 0.3, "data_driven", 8, True),
    ]
    results = run_ablation(
        configs=configs,
        steps=2,
        probe_items=4,
        output_dir=tmp_path,
        plot=False,
    )

    assert len(results) == len(configs)

    for res in results:
        csv_path = tmp_path / f"{res.run_id}.csv"
        assert csv_path.exists()
        rows = _read_rows(csv_path)
        assert rows
        last_row = rows[-1]
        # Basic sanity checks on numeric fields.
        assert float(last_row["recon_mse"]) > 0
        assert float(last_row["tokens_per_s"]) > 0
        assert 0.0 <= float(last_row["retrieval_top1"]) <= 1.0
