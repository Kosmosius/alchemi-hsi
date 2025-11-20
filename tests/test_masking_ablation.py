from __future__ import annotations

import csv
from pathlib import Path

import torch

from alchemi.models import MaskedAutoencoder, MaskingConfig


def _train_one(cfg: MaskingConfig, run_dir: Path, epochs: int = 100) -> float:
    torch.manual_seed(0)
    # Tiny synthetic "spectra": (B=1, T=4, C=8)
    data = torch.stack([torch.linspace(0, 1, 8) for _ in range(4)], dim=0).unsqueeze(0)

    model = MaskedAutoencoder(
        embed_dim=data.shape[-1],
        out_dim=data.shape[-1],
        mask_cfg=cfg,
    )
    opt = torch.optim.Adam(model.parameters(), lr=0.05)

    losses: list[float] = []
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        out = model(data, persist_dir=run_dir, include_unmasked_loss=False)
        out.loss.backward()
        opt.step()
        losses.append(float(out.loss.item()))
    return losses[-1]


def test_masking_ablation_better(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"

    baseline_cfg = MaskingConfig(
        spatial_mask_ratio=0.0,
        spectral_mask_ratio=0.8,
        mask_seed=0,
    )
    combined_cfg = MaskingConfig(
        spatial_mask_ratio=0.1,
        spectral_mask_ratio=0.1,
        spectral_grouping=2,
        mask_seed=1,
    )

    baseline_loss = _train_one(baseline_cfg, outputs / "run-baseline")
    combined_loss = _train_one(combined_cfg, outputs / "run-combined")

    csv_path = outputs / "ablation.csv"
    outputs.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["config", "loss"])
        writer.writeheader()
        writer.writerow({"config": "spectral-only", "loss": baseline_loss})
        writer.writerow({"config": "spatial+spectral", "loss": combined_loss})

    # Combined spatial+spectral should win.
    assert combined_loss < baseline_loss, "combined masking should improve reconstruction"

    # Mask configs should be persisted for both runs.
    for run_name in ("run-baseline", "run-combined"):
        cfg_paths = list((outputs / run_name).rglob("mask_config.json"))
        assert cfg_paths, f"mask config not persisted for {run_name}"
