import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

seed_labels = importlib.import_module("tools.seed_labels")


class DummyCompleted:
    def __init__(self, stdout: str = "", stderr: str = "") -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = 0
        self.args = []


def write_labels(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_dry_run_prints_intended_changes(monkeypatch, tmp_path, capsys):
    labels_path = write_labels(
        tmp_path,
        {
            "labels": [
                {"name": "area:new", "color": "abcdef", "description": "Example"},
            ]
        },
    )

    def fake_gh_command(args):
        assert args[:2] == ["label", "list"]
        return DummyCompleted(stdout=json.dumps([]))

    monkeypatch.setattr(seed_labels, "gh_command", fake_gh_command)
    monkeypatch.setenv("GH_TOKEN", "placeholder")

    exit_code = seed_labels.main(["--labels-path", str(labels_path), "--dry-run"])
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "Would create label 'area:new'" in captured.out


def test_skips_when_no_token(monkeypatch, tmp_path, capsys):
    labels_path = write_labels(
        tmp_path,
        {
            "labels": [
                {"name": "area:existing", "color": "123456", "description": ""},
            ]
        },
    )

    def fail_if_called(args):  # pragma: no cover - defensive
        raise AssertionError("gh_command should not be invoked when GH_TOKEN is missing")

    monkeypatch.setattr(seed_labels, "gh_command", fail_if_called)
    monkeypatch.delenv("GH_TOKEN", raising=False)

    exit_code = seed_labels.main(["--labels-path", str(labels_path)])
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "GH_TOKEN is not set" in captured.out


def test_creates_missing_labels(monkeypatch, tmp_path):
    labels_path = write_labels(
        tmp_path,
        {
            "labels": [
                {"name": "area:platform", "color": "0052cc", "description": "Infra"},
            ]
        },
    )

    calls = []

    def fake_gh_command(args):
        calls.append(list(args))
        if args[:2] == ["label", "list"]:
            return DummyCompleted(stdout=json.dumps([]))
        if args[:2] == ["label", "create"]:
            return DummyCompleted()
        raise AssertionError(f"Unexpected gh command: {args}")

    monkeypatch.setattr(seed_labels, "gh_command", fake_gh_command)
    monkeypatch.setenv("GH_TOKEN", "placeholder")

    exit_code = seed_labels.main(["--labels-path", str(labels_path)])
    assert exit_code == 0

    assert [
        "label",
        "create",
        "area:platform",
        "--color",
        "0052CC",
        "--description",
        "Infra",
    ] in calls
