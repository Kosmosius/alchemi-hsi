# Experiment configuration guide

Alchemi uses a single, shared source of truth for top-level runtime settings and
module-specific schemas for each trainer. The ``src/alchemi/config.py``
``RuntimeConfig`` dataclass is the canonical schema for experiment-wide knobs
(seed, device, dtype, AMP). Training entrypoints merge the ``global`` block from
YAML into this schema before wiring trainer-specific options.

## Canonical runtime schema

```yaml
# Top-level block shared by all trainers
global:
  seed: 42              # applied to CPU/NumPy/PyTorch
  device: auto          # "auto" selects CUDA when available, otherwise CPU
  dtype: float32        # default torch dtype
  amp_dtype: bf16       # optional autocast dtype when AMP is enabled
  deterministic: false  # request deterministic algorithms when available
```

``RuntimeConfig`` lives in ``src/alchemi/config.py`` and is imported by both the
MAE harness and the alignment trainer. CLI commands surface ``--seed`` to
override the YAML without editing files.

## Trainer-specific schemas

* **Synthetic MAE harness** — ``src/alchemi/training/config.py`` defines the
  Pydantic ``TrainCfg`` model used by ``run_pretrain_mae``. Its YAML lives under
  ``configs/train.mae.yaml`` with fields nested in a ``train`` mapping
  (embedding dim, masking ratios, logging cadence, etc.).
* **Alignment trainer (mainline)** — ``src/alchemi/train/alignment_trainer.py``
  declares the ``AlignmentExperimentConfig`` dataclass and nested sections for
  ``trainer``, ``data``, ``tokenizer``, ``model``, ``optimizer``, ``loss``,
  ``cycle``, and optional ``banddepth``. The reference config is
  ``configs/phase2/alignment.yaml``.

Trainer schemas intentionally mirror their YAML structure to keep authoring
simple. Add new knobs directly to the relevant dataclass or Pydantic model so
validation stays centralized.

## Authoring a new config

1. Start from the closest YAML exemplar in ``configs/`` (e.g.,
   ``configs/phase2/alignment.yaml`` or ``configs/train.mae.yaml``).
2. Keep shared runtime settings in the top-level ``global`` block; do **not**
   invent new per-trainer keys for seeds/devices/dtypes.
3. Add trainer-specific options under the appropriate section and extend the
   corresponding schema (``TrainCfg`` or ``AlignmentExperimentConfig``) if a new
   field is required.
4. Run the trainer via the CLI (`alchemi pretrain-mae` or `alchemi align train`)
   to validate the file; schema mismatches are surfaced as friendly errors.

``configs/schema.py`` previously held an example Pydantic model but is no longer
used; refer to the modules above for the authoritative schemas.
