from __future__ import annotations

import csv
from pathlib import Path

from typer.testing import CliRunner

from alchemi import cli
from alchemi.models.mae_baselines import BASELINES, available_baselines


def test_mae_cli_exposes_baseline_flags() -> None:
    runner = CliRunner()
    result = runner.invoke(cli.app, ["pretrain-mae", "--help"])

    assert result.exit_code == 0
    assert "--no-spatial-mask" in result.stdout
    assert "--no-posenc" in result.stdout


def test_available_baselines_and_csv_agree() -> None:
    models_from_registry = {b.model for b in available_baselines()}
    assert models_from_registry == {
        "mae_main",
        "mae_no_spatial_mask",
        "mae_no_posenc",
    }

    path = Path("data/ablation_mae.csv")
    rows = list(csv.DictReader(path.read_text().splitlines()))

    models_from_csv = {row["model"] for row in rows}
    assert models_from_registry.issubset(models_from_csv)

    steps_by_model = {row["model"]: int(row["steps"]) for row in rows}
    main_steps = steps_by_model["mae_main"]
    for name in ("mae_no_spatial_mask", "mae_no_posenc"):
        assert steps_by_model[name] == main_steps

    scores = {row["model"]: float(row["score"]) for row in rows}
    assert scores["mae_main"] < scores["mae_no_spatial_mask"]
    assert scores["mae_main"] < scores["mae_no_posenc"]
