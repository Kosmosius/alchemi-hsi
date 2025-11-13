# Definition of Done

All pull requests must satisfy the following checklist before merging. The checklist is referenced from the [PR template](../.github/PULL_REQUEST_TEMPLATE.md) and should be treated as a contract for contributors and reviewers.

## Quality gates

- **Lint clean** — `make lint` passes with no warnings (Black formatting + Ruff lint). If the Make target is not yet available, run `black --check src tests` and `ruff check src tests` manually.
- **Type safe** — `make typecheck` succeeds. The initial implementation will wrap `pyright`; until it lands, contributors must run the equivalent type checker locally and resolve all errors.
- **Unit tests** — `make test` (pytest) passes on a clean working tree. Tests must cover both success and failure modes.
- **Coverage** — `make coverage` meets or exceeds the project threshold (80% line coverage for the touched modules). Include coverage reports in the PR description when the automation is unavailable.

## Documentation and contracts

- **Docstrings** — Public functions, classes, and CLI commands include docstrings with:
  - Purpose in one sentence.
  - Parameters annotated with types, expected units, and valid ranges.
  - Return value description and units.
  - Exceptions raised and their trigger conditions.
- **Changelog** — Meaningful user-facing changes are recorded in the appropriate docs (README, module guides, or release notes) during Phase 1. The PR should explain where the change was documented.
- **Quickstarts** — If the feature changes the CLI or user workflow, update the relevant guide under `docs/quickstarts/`.

## Reproducibility

- **Deterministic seeds** — Use `alchemi.training.seed.seed_everything` (or the forthcoming equivalent for new modules) in scripts, tests, and notebooks that rely on randomness.
- **Configuration capture** — Any new pipeline entry point must persist the config it consumed (e.g., copy `yaml` files next to outputs or log the resolved config).
- **Data provenance** — Tests and examples must state the sensor, version, and acquisition metadata required to reproduce inputs. Where assets are too large for the repo, document the download script or source dataset ID.

## Review process

- At least one maintainer review is required.
- CI must run on the feature branch. If automation is temporarily unavailable, attach terminal output proving lint, type, test, and coverage gates passed locally.
- No TODOs or commented-out code remain, unless tracked by a follow-up issue referenced in the PR.
