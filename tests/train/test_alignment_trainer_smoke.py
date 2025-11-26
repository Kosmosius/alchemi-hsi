from __future__ import annotations

import math

import numpy as np
import torch

from alchemi.align.testing import SyntheticAlignmentDataset
from alchemi.train.alignment_trainer import (
    AlignmentExperimentConfig,
    AlignmentTrainer,
    CycleConfig,
    DataConfig,
    LabGridConfig,
    LossConfig,
    ModelConfig,
    OptimizerConfig,
    TokenizerConfig,
    TrainerConfig,
    _TensorBatch,
)
from alchemi.tokens.band_tokenizer import BandTokConfig


class _SyntheticAlignmentTrainer(AlignmentTrainer):
    """AlignmentTrainer that pulls batches from a deterministic synthetic dataset."""

    def __init__(self, config: AlignmentExperimentConfig, dataset: SyntheticAlignmentDataset):
        self._dataset = dataset
        self._cursor = 0
        super().__init__(config)

    def _build_batch(self, batch_size: int) -> _TensorBatch:  # type: ignore[override]
        indices = [(self._cursor + idx) % len(self._dataset) for idx in range(batch_size)]
        self._cursor = (self._cursor + batch_size) % len(self._dataset)
        lab_values, sensor_values = self._dataset.batch(indices)

        lab_tokens, lab_masks, sensor_tokens, sensor_masks = [], [], [], []
        for lab_row, sensor_row in zip(lab_values, sensor_values):
            lab_tok = self.tokenizer(
                lab_row,
                self.lab_wavelengths,
                axis_unit=self.axis_unit,
                width=self.lab_fwhm,
            )
            sensor_tok = self.tokenizer(
                sensor_row,
                self._dataset.sensor_wavelengths_nm,
                axis_unit=self.axis_unit,
            )
            lab_tokens.append(lab_tok.bands)
            sensor_tokens.append(sensor_tok.bands)
            lab_masks.append(~lab_tok.meta.invalid_mask)
            sensor_masks.append(~sensor_tok.meta.invalid_mask)

        lab_tokens_tensor = self._to_tensor(np.stack(lab_tokens))
        lab_mask_tensor = self._to_tensor(np.stack(lab_masks), dtype=torch.bool)
        sensor_tokens_tensor = self._to_tensor(np.stack(sensor_tokens))
        sensor_mask_tensor = self._to_tensor(np.stack(sensor_masks), dtype=torch.bool)
        lab_values_tensor = self._to_tensor(lab_values)
        sensor_values_tensor = self._to_tensor(sensor_values)
        sensor_wavelengths = self._to_tensor(self._dataset.sensor_wavelengths_nm)
        lab_wavelengths = self._to_tensor(self.lab_wavelengths)

        return _TensorBatch(
            lab_tokens=lab_tokens_tensor,
            lab_mask=lab_mask_tensor,
            sensor_tokens=sensor_tokens_tensor,
            sensor_mask=sensor_mask_tensor,
            lab_values=lab_values_tensor,
            sensor_values=sensor_values_tensor,
            sensor_wavelengths=sensor_wavelengths,
            lab_wavelengths=lab_wavelengths,
        )

    def _to_tensor(self, array: np.ndarray, dtype: torch.dtype | None = None):
        tensor = torch.as_tensor(array, device=self.device)
        if dtype is None:
            return tensor.to(dtype=self.dtype)
        return tensor.to(dtype=dtype)


def test_alignment_trainer_synthetic_smoke():
    dataset = SyntheticAlignmentDataset.create(num_pairs=48, num_samples=48, seed=7)

    config = AlignmentExperimentConfig(
        trainer=TrainerConfig(
            batch_size=6,
            max_steps=5,
            log_every=1,
            eval_every=2,
            device="cpu",
            dtype="float32",
            seed=123,
            deterministic=True,
            grad_clip_norm=1.0,
            use_amp=False,
        ),
        data=DataConfig(
            lab_grid_nm=LabGridConfig(
                start=float(dataset.lab_wavelengths_nm.min()),
                stop=float(dataset.lab_wavelengths_nm.max()),
                num=dataset.lab_wavelengths_nm.size,
            ),
            lab_fwhm_nm=5.0,
            lab_noise_std=0.0,
            synthetic_peaks=0,
            sensor_noise_rel=0.0,
        ),
        tokenizer=TokenizerConfig(
            axis_unit=dataset.axis_unit,
            params=BandTokConfig(n_fourier_frequencies=8, axis_norm="zscore"),
        ),
        model=ModelConfig(embed_dim=12, depth=1, heads=2, cycle_weight=0.0),
        optimizer=OptimizerConfig(lr=5e-3, weight_decay=0.0),
        loss=LossConfig(tau_init=0.07, learnable_tau=True, gather_ddp=False),
        cycle=CycleConfig(enabled=False),
    )

    trainer = _SyntheticAlignmentTrainer(config, dataset)
    history = trainer.train(max_steps=config.trainer.max_steps)

    losses = [entry["loss"] for entry in history]
    assert all(math.isfinite(value) for value in losses), "loss contains non-finite values"
    assert losses[-1] <= losses[0] + 0.2, "loss diverged on synthetic data"

    eval_batch = trainer._build_batch(batch_size=6)
    metrics = trainer._evaluate(eval_batch)
    assert math.isfinite(metrics["spectral_delta"])
    assert -1e-3 <= metrics["spectral_delta"] <= math.pi
    assert 0.0 <= metrics["retrieval@1"] <= 1.0
    assert metrics["retrieval@1"] >= 1.0 / config.trainer.batch_size
