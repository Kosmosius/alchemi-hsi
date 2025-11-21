# ARCH

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
