"""Seed GitHub labels for the repository using the GitHub CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

LABEL_FILE_DEFAULT = Path(__file__).resolve().parent / "data" / "labels.json"


@dataclass
class Label:
    """A desired GitHub label definition."""

    name: str
    color: str
    description: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> Label:
        color = data.get("color") or label_color_from_name(data["name"])
        return cls(
            name=data["name"],
            color=color.upper(),
            description=data.get("description") or None,
        )


def label_color_from_name(name: str) -> str:
    """Derive a deterministic hex color for a label name.

    GitHub label colors are six-character hex strings. We take the first
    six characters of a SHA1 digest of the label name to keep the mapping
    stable across runs while avoiding a hard-coded palette.
    """

    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()
    return digest[:6].upper()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=LABEL_FILE_DEFAULT,
        help="Path to the JSON file describing the labels.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the changes that would be applied without modifying labels.",
    )
    return parser.parse_args(argv)


def load_desired_labels(path: Path) -> list[Label]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    raw_labels = payload.get("labels", [])
    return [Label.from_dict(item) for item in raw_labels]


def gh_command(args: Iterable[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["gh", *args],
        check=True,
        capture_output=True,
        text=True,
    )


def fetch_existing_labels() -> dict[str, Label]:
    result = gh_command(["label", "list", "--limit", "1000", "--json", "name,color,description"])
    data = json.loads(result.stdout or "[]")
    labels: dict[str, Label] = {}
    for entry in data:
        labels[entry["name"]] = Label(
            name=entry["name"],
            color=entry["color"].upper(),
            description=entry.get("description") or None,
        )
    return labels


@dataclass
class Action:
    kind: str
    label: Label


def calculate_actions(desired: Sequence[Label], existing: dict[str, Label]) -> list[Action]:
    actions: list[Action] = []
    for label in desired:
        current = existing.get(label.name)
        if current is None:
            actions.append(Action("create", label))
            continue
        if current.color != label.color or (current.description or "") != (label.description or ""):
            actions.append(Action("update", label))
    return actions


def apply_actions(actions: Sequence[Action], dry_run: bool) -> None:
    for action in actions:
        if dry_run:
            print(
                f"[DRY RUN] Would {action.kind} label '{action.label.name}' "
                f"(color={action.label.color}, description={action.label.description!r})"
            )
            continue

        if action.kind == "create":
            args = [
                "label",
                "create",
                action.label.name,
                "--color",
                action.label.color,
            ]
            if action.label.description:
                args.extend(["--description", action.label.description])
            gh_command(args)
        elif action.kind == "update":
            args = [
                "label",
                "edit",
                action.label.name,
                "--color",
                action.label.color,
            ]
            if action.label.description:
                args.extend(["--description", action.label.description])
            else:
                args.extend(["--description", ""])
            gh_command(args)
        else:
            raise ValueError(f"Unsupported action kind: {action.kind}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    labels = load_desired_labels(args.labels_path)

    if not labels:
        print(f"No labels defined in {args.labels_path}.")
        return 0

    gh_token = os.getenv("GH_TOKEN")
    if gh_token is None and not args.dry_run:
        print("GH_TOKEN is not set; skipping label seeding. Use --dry-run to preview changes.")
        return 0

    try:
        existing = fetch_existing_labels()
    except FileNotFoundError:
        print("GitHub CLI 'gh' is not installed or not found in PATH.", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        print(exc.stderr or exc.stdout, file=sys.stderr)
        return exc.returncode

    actions = calculate_actions(labels, existing)

    if not actions:
        if args.dry_run:
            print("[DRY RUN] No label changes required.")
        else:
            print("Labels are already up to date.")
        return 0

    apply_actions(actions, args.dry_run)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
