# Contributing to ALCHEMI-HSI

We welcome contributions that strengthen the hyperspectral ingestion, physics, and modeling stack. This guide explains how to get a development environment running, follow the project style, and prepare changes that meet our [Definition of Done](DEFINITION_OF_DONE.md).

## Development environment

1. **Clone the repository**
   ```bash
   git clone https://github.com/Kosmosius/alchemi-hsi.git
   cd alchemi-hsi
   ```
2. **Create and activate a Python 3.10 virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. **Install dependencies**
   Use the convenience target below to install the project in editable mode with developer extras and pre-commit hooks:
   ```bash
   make setup
   ```
   If you prefer to run the steps manually, execute the equivalent commands:
   ```bash
   pip install -e .[dev]
   pre-commit install
   ```
4. **Sync data assets (optional)**
   The quickstarts and tests rely on lightweight fixtures committed to the repo. For large scene downloads we will provide scripted helpers in Phase 1; see the issue tracker for ingest tasks.

## Project conventions

### Coding style

- Python code is formatted with **Black** using a 100 character line length and linted with **Ruff**. Run `make lint` (Ruff + Black check + isort check-only) before sending a change.
- Keep imports sorted and grouped (Black + Ruff will handle this automatically).
- Public APIs require docstrings that describe arguments, return values, units, and failure modes. See the [Definition of Done](DEFINITION_OF_DONE.md#documentation-and-contracts) for the required structure.
- Prefer `Path` objects over raw strings for filesystem interactions.

### Type hints

- All new or modified functions must be fully type annotated.
- Use `typing.Annotated` to record physical units when applicable.
- Run `make typecheck` to ensure the module type-checks cleanly with **mypy** (configured via `mypy.ini`).

### Tests and coverage

- Add or update unit tests in `tests/` for all new behaviors.
- Ensure `make test` (alias for `pytest`) and `make coverage` succeed locally before opening a pull request.
- Use `pytest --maxfail=1` during development for faster feedback.

### Commits and pull requests

- Use short-lived feature branches named `area/short-description` (e.g., `ingest/emit-loader`).
- Follow [Conventional Commits](https://www.conventionalcommits.org/) for commit messages (e.g., `feat(ingest): add EMIT L1B reader`).
- Rebase on `main` before opening a pull request to keep history linear.
- Every pull request must:
  - Reference the tracked issue(s).
  - Include a summary of changes, testing evidence, and any known limitations.
  - Check the boxes in the [PR template](../.github/PULL_REQUEST_TEMPLATE.md) confirming the Definition of Done is met.

#### Architecture Decision Records (ADRs)

Major architectural changes—including data contracts, tooling choices, and project policies—must include an ADR. Start from the [ADR template](adr/0000-template.md) and add the new record under `docs/adr/`. Reference the ADR in the pull request description alongside the code implementing the decision. See the [ADR index](DECISIONS.md) and [ADR guidance](adr/README.md) for the full workflow.

## Local CLI and notebooks

- The Typer CLI entry point lives at `alchemi.cli` and can be invoked with `python -m alchemi.cli` or the future console script `alchemi` once packaging is configured.
- Explore `notebooks/quickstart.ipynb` for an executable walkthrough built from seeded synthetic data. Add new exploratory notebooks under `notebooks/` and keep them deterministic by seeding randomness via `alchemi.training.seed.seed_everything`.

## Getting help

- File questions or feature proposals as GitHub issues and tag the relevant Phase/Area labels from `TASKS.yaml`.
- For urgent build or CI failures mention the `@Kosmosius/alchemi-maintainers` team in the issue.
