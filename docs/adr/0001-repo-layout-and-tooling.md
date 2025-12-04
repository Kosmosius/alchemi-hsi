# Repository layout, tooling, and CI

- Status: Accepted
- Date: 2025-12-04

## Context
ALCHEMI-HSI is a physics-aware, any-sensor hyperspectral foundation model. To support both the core library and experimental harnesses, the project needs a maintainable, testable, multi-package Python codebase that works well for library consumers and researchers. Agreeing on a standard layout, tooling stack, and CI workflow is essential to keep contributions consistent and reliable.

## Decision
- Use a `src/` layout with `alchemi` and `spectra` as separate installable packages.
- Build and package with hatchling.
- Format code with Black (line length 100) and sort imports with isort using the Black profile.
- Lint primarily with Ruff using the E, F, I, B, UP, NPY, RUF, and W rule families.
- Run mypy as the static type checker with strict defaults and targeted relaxations via `mypy.ini`.
- Test with `pytest` and `pytest-cov`.
- Use pre-commit locally to run Black, Ruff, isort, and basic checks.
- Use GitHub Actions (`ci.yml`) to run lint, type checks, and tests on Python 3.10 and 3.11 for every push and pull request.

## Consequences
### Positive
- Consistent style and quality controls across the project.
- Automated enforcement through pre-commit locally and CI in the repository.
- Easier onboarding because the single source of truth for tooling lives in `pyproject.toml`, `mypy.ini`, and `ruff.toml`.

### Negative / Tradeoffs
- Contributors must install and learn multiple tools (Black, Ruff, isort, mypy).
- CI covers several checks, adding some runtime to each run.
- Tight coupling to Python 3.10+ and PyTorch 2.2+ for development and testing.

## Alternatives Considered
- Using Pyright instead of mypy; retained mypy due to existing strict configuration and flexibility for per-module tuning.
- Using a flat package layout instead of the `src/` layout; rejected to avoid import shadowing and to keep installable packages explicit.
- Skipping pre-commit; rejected because local hooks help catch issues before CI.
- Using another CI provider or relying solely on local checks; GitHub Actions was chosen for integration with repository workflows and consistency across contributors.

## References
- [pyproject.toml](../../pyproject.toml)
- [mypy.ini](../../mypy.ini)
- [ruff.toml](../../ruff.toml)
- [GitHub Actions workflow](../../.github/workflows/ci.yml)
- [docs/CONTRIBUTING.md](../CONTRIBUTING.md)
- [docs/DEFINITION_OF_DONE.md](../DEFINITION_OF_DONE.md)
