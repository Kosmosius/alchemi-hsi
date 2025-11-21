# ALCHEMI — Atmosphere-aware Lab-to-scene Contrastive Hyperspectral Embeddings for Material Identity

[![CI](https://github.com/alchemi-hsi/alchemi-hsi/actions/workflows/ci.yml/badge.svg)](https://github.com/alchemi-hsi/alchemi-hsi/actions/workflows/ci.yml)

Physics-aware, SRF-aware, sensor-agnostic hyperspectral foundation model (SWIR/LWIR) for compound identification and unmixing.

```bash
pip install -e ".[dev]"
pre-commit install
pre-commit run --all-files
pytest -q                   # CPU-fast tests
python -m alchemi.cli --help
```

See docs/ for PRD, ARCH, DATA_SPEC, EVAL, ROADMAP, DECISIONS, APPENDIX. Phase‑1 quickstarts are collected under [`docs/quickstarts/`](docs/quickstarts/)—start with the [CLI walkthrough](docs/quickstarts/cli.md) to explore dataset validation and canonical cube exports end-to-end.

## Tiling large cubes

Process large scenes without loading the full raster into GPU/CPU memory by iterating over tiles:

```python
from alchemi.data.cube import Cube

cube = Cube(...)
for row_slice, col_slice, tile in cube.iter_tiles(tile_h=64, tile_w=64):
    logits = model(tile.data)  # run your model on a 64×64×C chip
    output[row_slice, col_slice] = logits
```

Edges are handled automatically, so partial tiles at the borders are yielded with the correct shape.

## Pretraining ablations

A lightweight synthetic harness in `scripts/ablate_pretrain.py` sweeps masking ratios, grouping modes/sizes, and any-sensor ingest. It emits one CSV per run plus reconstruction-MSE and throughput plots under `outputs/ablations/`, and reports a retrieval@1 sanity check over a toy probe set.

Run a small sweep (trimmed via `--max-runs` for smoke tests):

```bash
python scripts/ablate_pretrain.py run --steps 3 --max-runs 16
```

In the synthetic setup, a data-driven grouping with **G=8**, **mask_spatial=0.5**, **mask_spectral=0.3**, and **ingest_any=True** tends to minimize reconstruction MSE at comparable synthetic compute, so we use it as the recommended default when comparing pretraining variants.
