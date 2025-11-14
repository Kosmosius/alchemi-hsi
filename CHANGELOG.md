# Changelog

## Phase-1 complete (2025-11-14)

### Highlights
- Introduced a canonical hyperspectral `Cube` model, dataset sniffers, and Typer CLI commands for inspecting and exporting sensor data products (PRs #107, #108).
- Landed EMIT, EnMAP, AVIRIS-NG, HyTES, and MAKO ingestion pipelines spanning L1B radiance, L2S/L3 brightness temperatures, and mineral label helpers (PRs #67, #68, #70, #71, #88, #89, #95).
- Built a reusable spectral response function (SRF) registry with sensor-specific matrices, Gaussian fallbacks, and resampling utilities, plus normalized SRF assets for EMIT, EnMAP, AVIRIS-NG, HyTES, and MAKO (PRs #75–#80, #96).
- Added physics utilities for SWIR radiance↔reflectance conversion, Planck-law brightness temperature modeling, and atmospheric augmentation recipes with robust property-based tests (PRs #90–#94, #109, #110).
- Expanded documentation with Phase-1 quickstarts, canonical data model guidance, MAKO ingest walkthroughs, and workflow/process playbooks (PRs #66, #104, #105).

### Upgrade notes
- Regenerate SRF caches after pulling Phase-1 to ensure new JSON assets and registry entries are picked up.
- The new CLI commands require installing the project with developer extras (`pip install -e ".[dev]"`).
- Existing automation should switch to the consolidated `alchemi` Python package namespace introduced in Phase-1.
