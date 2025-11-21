## Experiment configuration and reproducibility

Training entrypoints consume YAML configs in ``configs/`` and share a small,
central schema defined in ``src/alchemi/config.py``. Each config can include a
``global`` block to standardize runtime settings:

```yaml
global:
  seed: 42
  device: auto          # "auto" selects CUDA when available, otherwise CPU
  dtype: float32        # e.g., float32, bf16, fp16
  amp_dtype: bf16       # optional autocast dtype when AMP is enabled
  deterministic: false
```

Both ``alchemi pretrain-mae`` and ``alchemi align train`` read these fields,
seed PyTorch/NumPy/CPU RNGs, set the default dtype, and log the effective
seed/device/dtype alongside the config path. CLI callers can override the seed
with ``--seed`` without editing the YAML.

Model- and data-specific hyperparameters stay in the existing sections (e.g.,
``train`` or ``trainer``). If ``global`` is missing, defaults are applied and
legacy keys like ``train.seed`` remain backward compatible.
## Training stacks

There are two training entrypoints in this repo:

* **Alignment trainer (mainline)** — `alchemi.train.alignment_trainer.AlignmentTrainer` and the
  `alchemi align train` CLI use the `BandTokenizer` and paired lab↔sensor spectra to train the
  encoder that is intended for downstream deployment. Configs live in `configs/phase2/alignment.yaml`
  and are dataclass-backed for easy YAML authoring.
* **Synthetic MAE harness (experimental)** — `alchemi.training.trainer.run_pretrain_mae` and the
  `alchemi pretrain-mae` CLI run a synthetic masked-autoencoder loop over random tokens. It uses
  `SpectralBasisProjector` plus a lightweight `MAEEncoder`/`MAEDecoder` pair to benchmark masking
  settings and throughput. Configs live in `configs/train.mae.yaml` and use the Pydantic
  `TrainCfg` schema.

The alignment trainer is the route to produce the encoder that ships in Phase-2 style experiments;
MAE pretraining remains available for ablations and quick performance measurements but is not the
source of the deployment encoder.

## Config layout

* `configs/phase2/alignment.yaml` — dataclass/YAML config consumed by `AlignmentTrainer.from_yaml`.
  Sections include `trainer` (loop settings), `data` (lab grid + synthetic sensor noise),
  `tokenizer` (band tokenization parameters), `model` (tower depth/width), and optional `cycle`/
  `banddepth` heads.
* `configs/train.mae.yaml` — Pydantic config used by `TrainCfg` for the MAE harness. Keys live under
  a top-level `train` mapping and control embedding size, masking ratios, and logging cadence.

## Encoder construction

Both training stacks share the same SetEncoder factory (`alchemi.models.factory.build_set_encoder`)
so changes to the pooling encoder automatically propagate to MAE and alignment runs. Tokenization is
stack-specific (`BandTokenizer` for alignment vs. `SpectralBasisProjector` for MAE), but the core
set encoder block is common.
