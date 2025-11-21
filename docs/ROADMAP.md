# ROADMAP

Status reflects the current dev branch; entries describe what exists today and
what remains planned.

## Phase 1 (implemented)
* Canonical data contracts (`Cube`, `Spectrum`, `Sample`, `SRFMatrix`) with
  monotonic wavelength validation and SRF row normalisation.
* Sensor ingest for EMIT, EnMAP, AVIRIS‑NG, HyTES, and Mako, including SI unit
  conversion, SRF tagging, and band masks.
* Physics helpers for Planck radiance↔BT, SWIR reflectance↔radiance, continuum
  removal, and band depth; sensor presets for EMIT/EnMAP/AVIRIS‑NG.
* Tokenisation + encoders: `BandTokenizer` defaults per sensor, shared
  set-encoder factory, optional SRF embeddings, and band‑depth head support.
* Alignment trainer (contrastive + optional cycle/band-depth) with AMP and seed
  control; synthetic MAE harness for masking ablations.
* Evaluation metrics for solids (macro‑F1, band‑depth MAE), gas PR/AUC+IoU,
  retrieval@k/spectral angle, and calibration (ECE + temperature scaling).

## Phase 2 (planned)
* Enriched gas heads (multi-gas joint models, plume morphology priors) and
  datasets beyond current benchmarks.
* Uncertainty: conformal prediction wrappers for retrieval/segmentation and
  tighter calibration reporting in evaluation scripts.
* Data expansion: additional LWIR/SWIR sensors and lab libraries wired through
  the same `Cube`/`Spectrum` contracts and SRF registry.
* Trainer polish: distributed alignment training on larger corpora, improved
  logging/checkpoint formats, and end-to-end ingestion/tokenisation pipelines for
  external callers.
