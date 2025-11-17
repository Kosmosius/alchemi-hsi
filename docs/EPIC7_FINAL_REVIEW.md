# EPIC #7 – Final Review Checklist (dev sync)

This log records the second and _final_ verification pass after syncing the work branch with the current `dev` tip (post-PR #124).  I re-ran the full validation procedure to ensure the "Any-Sensor Ingestion & Tokenization (SRF-optional)" epic still satisfies every acceptance criterion after the new merge.

## Review context
- **Git ref tested:** `2c620ec` (dev prior to this final review commit)
- **Commands executed:** `pytest -q`, `mypy`, and `python scripts/demo_any_sensor_tokenization.py --sensor emit --sensor enmap --sensor avirisng --sensor hytes`
- **Outcome:** All checks passed (aside from the expected rasterio/zarr warnings noted elsewhere) and the demo script confirmed variable-length tokens and SRF augmentation behave identically to the previous pass.

Each subsection below maps to the PRD requirements and documents the files/tests that make the behavior observable on the synced branch.

## A. Band tokenization API
- `alchemi.tokens.band_tokenizer.BandTokenizer` continues to expose the requested config fields (`BandTokConfig`) and supports value normalization, Fourier encoding, width defaults, and optional SRF embeddings.  The cm⁻¹/nm axis handling was manually checked via the demo script after the merge.
- `alchemi.tokens.registry` still provides sensor-aware presets and is used from the canonical `Cube.to_tokens()` pathway so that ingestion targets automatically inherit the tokenizer defaults.

## B. SRF-present path
- `alchemi.srf.utils` implements `default_band_widths` and `build_srf_band_embeddings` helpers that backfill widths and distill SRFs when available.
- SRF metadata lookups are validated by `tests/tokens/test_band_tokenizer_srf_present.py`, ensuring the embeddings materially affect the token outputs.  No regressions were observed after rerunning the suite on the synced branch.

## C. Synthetic sensor generator / SRF randomization
- `alchemi.srf.synthetic` produces deterministic SRF grids with normalized rows; `RandomSensorProject` in `alchemi.align.transforms` projects lab spectra through those grids for training-time augmentation.  The demo script exercised the projection step again to double-check numerical stability.
- `tests/srf/test_synthetic_srf_randomization.py` plus `tests/align/test_random_sensor_project.py` verify SRF normalization, deterministic seeds, and projection fidelity.

## D. Integration with ingest + trainer
- Canonical cube ingestion paths expose axis metadata and `Cube.to_tokens()` so every ingest (EMIT, EnMAP, AVIRIS-NG, HyTES, SPLIB) produces variable-length token sequences.  The latest `dev` changes did not disturb these plumbing layers (confirmed via CLI smoke tests and the demo script).
- `tests/integration/test_any_sensor_tokenization.py` exercises ingest → Cube → tokenizer for the supported sensors, confirming monotonic axes and correct band counts.

## E. Tests and regression coverage
- `tests/tokens/test_band_tokenizer_shapes.py` ensures nm and cm⁻¹ pathways stay numerically stable under shuffling.
- The full pytest suite (117 tests) passes (aside from expected rasterio/zarr warnings), confirming compatibility after the merges leading up to PR #124.  `mypy` remains clean on the tokenizer + align packages.
