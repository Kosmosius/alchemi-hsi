# EVAL

Evaluation is implemented in `alchemi.eval` for encoder outputs and in
`alchemi.training.metrics`/`alchemi.training.calibration` for auxiliary checks.
The default assumptions mirror the alignment trainer: paired lab↔sensor spectra
producing L2-normalised embeddings.

## Metrics

* **Macro‑F1** — `eval.metrics_solids.macro_f1` computes macro‑averaged F1 on
  discrete class labels (e.g., mineral IDs) using scikit‑learn.
* **Band‑depth MAE** — `eval.metrics_solids.banddepth_mae` evaluates mean
  absolute error of continuum‑removed band depths across configured windows using
  `physics.swir.band_depth`.
* **Spectral angle** — `eval.retrieval.spectral_angle_deltas` measures mean
  spectral angle for matched vs. mismatched lab/sensor embeddings; standalone
  `training.metrics.spectral_angle` supports raw spectra comparisons.
* **Retrieval@k** — `eval.retrieval.retrieval_at_k`/`retrieval_summary` compute
  recall@k for paired embeddings; `compute_retrieval_at_k` additionally reports
  precision.
* **Gas detection** — `eval.metrics_gas.pr_auc_iou` returns average precision and
  IoU at a supplied threshold for pixelwise score maps.
* **Calibration** — `training.metrics.ece_score` computes expected calibration
  error from confidences and correctness flags; `training.calibration.TemperatureScaler`
  fits a temperature to minimise ECE.

## Pipelines

### Solids (reflectance/radiance)
`eval.evaluate_solids` consumes ground-truth labels and predictions, optionally
with wavelength grids and reflectance arrays to also report band‑depth MAE. A
minimal call: `evaluate_solids(y_true, y_pred)` → `{"macro_f1": …}`. Providing
`nm`, `R_true`, `R_pred`, and `windows` appends `banddepth_mae`.

### Gas benchmarks
`eval.evaluate_gases` wraps `pr_auc_iou(mask_true, score_map, thresh=0.5)` to
return `{"ap": …, "iou": …}` for binary plume maps. Thresholds are explicit so
runs can sweep operating points.

### Alignment retrieval
For alignment checkpoints, embeddings from lab and sensor towers feed
`evaluate_alignment`, which reports retrieval@k via cosine similarity. The
helper `spectral_angle_deltas` in `eval.retrieval` can be called separately to
inspect separation between matched and mismatched pairs.

### Calibration checks
When heads emit logits (e.g., classification ablations),
`TemperatureScaler.fit` followed by `transform` can be applied before computing
ECE/NLL. The scaler falls back to identity when optimisation fails, so it is safe
in automated sweeps.
