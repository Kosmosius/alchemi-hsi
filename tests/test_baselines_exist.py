from __future__ import annotations

import csv
import inspect
from pathlib import Path

from typer.testing import CliRunner

from alchemi import cli
from alchemi.models.mae_baselines import available_baselines


def test_mae_cli_exposes_baseline_flags() -> None:
    runner = CliRunner()
    result = runner.invoke(cli.app, ["pretrain-mae", "--help"])

    assert result.exit_code == 0

    pretrain_cmd = None
    for command in cli.app.registered_commands:
        if command.callback.__name__ == "pretrain_mae":
            pretrain_cmd = command
            break

    assert pretrain_cmd is not None, "pretrain_mae command is not registered on cli.app"

    signature = inspect.signature(pretrain_cmd.callback)
    parameters = signature.parameters

    assert "no_spatial_mask" in parameters
    assert "no_posenc" in parameters


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
