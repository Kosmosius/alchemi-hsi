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
