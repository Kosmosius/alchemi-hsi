# Architecture Outline

The repository follows a `src/` layout with the reusable `alchemi` library separated from runnable entrypoints and configuration:

- `src/alchemi/`: Library package with physics, data adapters, models, training loops, evaluation utilities, registries, and command-line interfaces.
- `configs/`: Hydra-style configuration stubs for datasets, models, training stages, and experiments.
- `scripts/`: Thin wrappers that call into `alchemi.cli` for training, evaluation, preprocessing, and analysis tools.
- `tests/`: Unit and integration tests that will enforce the physics and data contracts from the design document.
- `docs/`: Documentation and design assets, including `docs/design/alchemi_design.tex` as the authoritative specification.
- `resources/`: Small assets such as SRFs or toy datasets used for testing and examples.
- `notebooks/`: Optional exploratory analyses or prototypes.

This outline will be refined as modules mature and as the design doc is reflected in code.
