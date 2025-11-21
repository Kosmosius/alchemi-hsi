# ARCH

This repository is organised around a single canonical cube representation that
feeds both training stacks (alignment and synthetic MAE) and downstream
evaluation utilities. The current dev branch implements the following flow:

```
Sensor products → ingest (sensor-specific) → Cube/Sample → physics helpers
                   ↓                                      ↓
            BandTokenizer / SpectralBasisProjector → Set encoder → heads
                                                       ↓
                                              eval metrics + exports
```

## Module layout and data flow

* **Ingest** (`alchemi.data.io.*` + `alchemi.ingest.*`) normalises EMIT, EnMAP,
  AVIRIS‑NG, HyTES, and Mako products. The xarray readers harmonise units and
  masks; the ingest helpers lift them into canonical `Cube` instances
  (`alchemi.data.cube.Cube`) and optionally tag SRF identifiers.
* **Canonical types** (`alchemi.types`) define `WavelengthGrid` (monotonic nm),
  `Spectrum` (1‑D values + mask), `Sample` (per‑pixel spectrum + metadata), and
  `SRFMatrix` (per‑band spectral response with trapezoidal normalisation).
  Cubes expose `sample_at` to produce `Sample` objects for lab ↔ sensor pairing.
* **Physics helpers** (`alchemi.physics.*`) perform sensor‑agnostic conversions:
  Planck radiance↔BT (`physics.planck`), SWIR reflectance↔radiance and continuum
  removal/band depth (`physics.swir`), and sensor presets (`physics.swir_emit`,
  `physics.swir_enmap`, `physics.swir_avirisng`). These helpers are used inside
  ingest, tokenisation, and evaluation utilities.
* **Tokenisation**
  * **Alignment path:** `tokens.band_tokenizer.BandTokenizer` turns banded
    spectra into ordered token sets (value + wavelength + optional FWHM/SRF
    embeddings). Defaults come from `tokens.registry.get_default_tokenizer`
    using sensor IDs.
  * **MAE path:** `models.SpectralBasisProjector` converts spectra to basis
    coefficients for synthetic masking experiments.
* **Encoders and heads**
  * Both stacks share `models.factory.build_set_encoder` to build the pooling
    encoder tower.
  * Alignment attaches contrastive heads (`align.losses.info_nce_symmetric`),
    optional cycle reconstruction (`align.cycle.CycleReconstructionHeads`), and
    an optional `heads.banddepth.BandDepthHead` for supervised band‑depth
    regression.
  * MAE uses lightweight `models.MAEEncoder`/`MAEDecoder` over basis tokens and
    a reconstruction loss (`losses.ReconstructionLoss`).
* **Evaluation** (`alchemi.eval`) consumes encoder outputs to compute retrieval,
  spectral angle deltas, band‑depth MAE, macro‑F1, gas AP/IoU, and calibration
  metrics (ECE).

## Training stacks

### Alignment trainer (mainline)
* Entrypoint: `alchemi.align train` → `train/alignment_trainer.AlignmentTrainer`.
* Data: paired lab spectra on a configurable uniform grid plus synthetic sensor
  perturbations (`align.batch_builders.build_emit_pairs` and related helpers).
* Tokeniser: `BandTokenizer` configured via `configs/phase2/alignment.yaml`
  (`AlignmentExperimentConfig.from_yaml`). SRF embeddings can be injected when
  available.
* Model: shared set encoder tower with projection heads for lab and sensor
  branches. Optional cycle and band‑depth heads are enabled through the YAML.
* Losses/metrics: symmetric InfoNCE with learnable temperature, retrieval@k and
  spectral‑angle deltas during eval, and optional band‑depth/cycle losses.
* AMP/precision: configurable via the `global`/`trainer` `device`, `dtype`, and
  `amp_dtype` settings.

### Synthetic MAE harness (experimental)
* Entrypoint: `alchemi pretrain-mae` → `training.trainer.run_pretrain_mae` using
  the `TrainCfg` Pydantic schema (`configs/train.mae.yaml`).
* Data: random spectra projected by `SpectralBasisProjector`, masked
  spectrally/spatially according to `MaskingConfig`.
* Model: `MAEEncoder`/`MAEDecoder` over basis tokens plus optional
  `BandDepthHead` and `DomainDiscriminator` for auxiliary losses.
* Losses/metrics: reconstruction loss, InfoNCE variants, spectral smoothness,
  throughput statistics; primarily for ablations, not deployment.

## Configuration and reproducibility

Training entrypoints consume YAML configs in `configs/` and share a small
central schema defined in `src/alchemi/config.py`. Each config can include a
`global` block to standardise runtime settings:

```yaml
global:
  seed: 42
  device: auto          # "auto" selects CUDA when available, otherwise CPU
  dtype: float32        # e.g., float32, bf16, fp16
  amp_dtype: bf16       # optional autocast dtype when AMP is enabled
  deterministic: false
```

Both `alchemi pretrain-mae` and `alchemi align train` read these fields, seed
PyTorch/NumPy/CPU RNGs, set the default dtype, and log the effective
seed/device/dtype alongside the config path. CLI callers can override the seed
with `--seed` without editing the YAML.

Model- and data-specific hyperparameters stay in the existing sections (e.g.,
`train` or `trainer`). If `global` is missing, defaults are applied and legacy
keys like `train.seed` remain backward compatible.

### Config layout

* `configs/phase2/alignment.yaml` — dataclass/YAML config consumed by
  `AlignmentTrainer.from_yaml`. Sections include `trainer` (loop settings),
  `data` (lab grid + synthetic sensor noise), `tokenizer` (band tokenisation
  parameters), `model` (tower depth/width), and optional `cycle`/`banddepth`
  heads.
* `configs/train.mae.yaml` — Pydantic config used by `TrainCfg` for the MAE
  harness. Keys live under a top-level `train` mapping and control embedding
  size, masking ratios, and logging cadence.

### Shared encoder construction

Both training stacks share the same set-encoder factory
(`alchemi.models.factory.build_set_encoder`) so changes to the pooling encoder
automatically propagate to MAE and alignment runs. Tokenisation is
stack-specific (`BandTokenizer` for alignment vs. `SpectralBasisProjector` for
MAE), but the core encoder block is common.
