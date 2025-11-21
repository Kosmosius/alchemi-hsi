from __future__ import annotations

import re

from typer.testing import CliRunner

from alchemi.cli import app


def test_validate_data_missing_file() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["validate-data", "--config", "does-not-exist.yaml"])

    assert result.exit_code == 1
    combined_output = result.stdout + (result.stderr or "")
    assert "does-not-exist.yaml" in combined_output
    assert "Traceback" not in combined_output


def test_version_command_reports_details() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert re.search(r"\d+\.\d+\.\d+", result.stdout)
    assert "Supported sensors" in result.stdout
    assert "Canonical cubes" in result.stdout
