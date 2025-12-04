# Repository layout, tooling, and CI

**Status:** Accepted

**Date:** 2025-05-11

## Context

ALCHEMI-HSI is a physics-aware, any-sensor hyperspectral foundation model aimed at supporting research and applications across diverse sensing modalities. The project needs a maintainable, testable, multi-package Python codebase that accommodates the main library and experimental harness while staying approachable for contributors. Establishing a shared repository layout, tooling stack, and continuous integration (CI) pipeline ensures consistency and reliability as the project grows.

## Decision

- Adopt a `src/` layout with two installable packages: `alchemi` for the core library and `spectra` for the SpectralEarth/Multi-spectral experimental harness.
- Use hatchling as the build backend to package both namespaces via the wheel target configuration.
- Enforce formatting with Black (line length 100) and isort using the Black profile and matching line length, treating `alchemi` and `spectra` as first-party modules.
- Use Ruff as the primary linter, enabling rule families E/F/I/B/UP/NPY/RUF/W with Python 3.10/3.11 targets and line length 100.
- Use mypy as the static type checker with strict defaults and module-specific relaxations configured in `mypy.ini` to balance rigor with practicality.
- Run tests with pytest and coverage via pytest-cov to validate functionality and measure test coverage.
- Use pre-commit to run formatting, linting, and basic hygiene checks locally before pushes.
- Use GitHub Actions (`ci.yml`) to run linting, formatting checks, type checking, and tests on Python 3.10 and 3.11 for every push and pull request.

## Consequences

- Positive:
  - Consistent coding style and imports across the project.
  - Automated enforcement through pre-commit and CI reduces regressions and review burden.
  - Single-source tooling configuration in `pyproject.toml`, `mypy.ini`, and `ruff.toml` simplifies onboarding and local setup.
- Negative / tradeoffs:
  - Contributors must install and learn multiple tools (Black, Ruff, isort, mypy).
  - CI runs comprehensive checks, modestly increasing runtime for each push/PR.
  - Tight coupling to Python 3.10+ and PyTorch 2.2+ constrains older environments.

## Alternatives Considered

- Using pyright instead of mypy: rejected because mypy was already configured with strict settings and allows granular per-module tuning via `mypy.ini`.
- Using a flat package layout instead of `src/`: rejected to avoid import path leaks and to better isolate installed packages from the repository root.
- Skipping pre-commit hooks: rejected because local automation reduces style drift and lowers CI friction.
- Using a different CI provider or relying only on local checks: rejected because GitHub Actions is already integrated and provides consistent, repeatable validation on every push/PR.

## References

- [`pyproject.toml`](../../pyproject.toml)
- [`mypy.ini`](../../mypy.ini)
- [`ruff.toml`](../../ruff.toml)
- [`docs/CONTRIBUTING.md`](../CONTRIBUTING.md)
- [`docs/DEFINITION_OF_DONE.md`](../DEFINITION_OF_DONE.md)
- [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml)
