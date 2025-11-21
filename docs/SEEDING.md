# GitHub seeding guide

This repository includes helper scripts for seeding GitHub labels, milestones, epics, and issues from YAML. Use this document to understand the schema expected by `TASKS.yaml` and how label colors are derived.

## Deterministic label colors

Both `scripts/seed_github.sh` and `tools/seed_labels.py` derive label colors from the label name when a color is not explicitly provided. The first six hex characters of the SHA1 digest of the label name are used, upper‑cased (e.g., `priority:P1` → `8DC8F9`). Rerunning the scripts yields the same color for a given label name.

## `TASKS.yaml` schema

`TASKS.yaml` is organized into a few top‑level sections:

- `version`: Integer schema version.
- `repo`: `owner/repo` slug for the target repository.
- `project`:
  - `name`: Name of the project board (if used elsewhere).
  - `fields`: List of project fields.
- `labels`: A list of label names. Colors are auto‑derived as described above unless you seed labels via `tools/seed_labels.py` with explicit colors.
- `milestones`: A list of milestone objects with:
  - `key` (required): Short identifier used when linking epics/issues to milestones.
  - `title` (required): GitHub milestone title.
  - `due` (optional): `YYYY-MM-DD` due date.
- `epics` and `issues`: Lists of issue definitions with the same shape. Each entry supports:
  - `key` (recommended): Short identifier for cross‑referencing.
  - `title` (required): Issue title. A `PREFIX` passed to `seed_group` in the script is prepended when seeding epics.
  - `body` (optional): Markdown description.
  - `labels` (optional): List of label names applied to the issue.
  - `milestone` (optional): `key` of an entry in `milestones` to attach during creation.
  - `acceptance` (optional): List of bullet points rendered under an "Acceptance criteria" heading.
  - `tests` (optional): List of bullet points rendered under a "Tests" heading.

The schema is backwards compatible with existing `TASKS.seed.yaml`; new fields are optional unless marked required above.

## Seeding entry points

- `scripts/seed_github.sh owner/repo [TASKS.yaml]`: Shell helper that seeds labels (with deterministic colors), milestones, epics, and issues using `gh`, `yq`, and `jq`.
- `tools/seed_labels.py`: Python helper for seeding labels from `tools/data/labels.json` or another JSON file. Missing colors are derived deterministically from names.

Both scripts support dry runs (`seed_labels.py --dry-run`) or can be run against forks/temporary repos when testing changes.
