"""End-to-end SRF-blind vs SRF-aware robustness experiment.

This toy experiment trains a tiny ingest + head model in SRF-aware mode and then
measures the performance drop when re-running evaluation in SRF-blind mode. It
follows the terminology in Section 6 of the design doc: SRF-aware mode uses
per-band SRF embeddings, while SRF-blind mode disables them but keeps all other
inputs identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import yaml
from torch import Tensor, nn

from alchemi.data.datasets import SyntheticSensorDataset
from alchemi.spectral import Sample, Spectrum
from alchemi.srf.synthetic import SyntheticSensorConfig
from alchemi.types import QuantityKind


@dataclass(slots=True)
class SrfBlindVsAwareConfig:
    """Configuration for the SRF-blind vs SRF-aware robustness experiment."""

    epochs: int = 8
    learning_rate: float = 5e-2
    hidden_dim: int = 32
    embed_dim: int = 32
    num_samples: int = 6
    highres_points: int = 256
    seed: int = 7
    sensor_bands: int = 12
    fwhm_range_nm: tuple[float, float] = (8.0, 18.0)


def load_srf_blind_vs_aware_config(path_or_mapping: str | Path | Mapping[str, object] | None) -> SrfBlindVsAwareConfig:
    """Load configuration from a mapping or YAML file."""

    if path_or_mapping is None:
        return SrfBlindVsAwareConfig()
    if isinstance(path_or_mapping, Mapping):
        return SrfBlindVsAwareConfig(**path_or_mapping)

    cfg_path = Path(path_or_mapping)
    with cfg_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        raise TypeError(f"Expected mapping in {cfg_path}, found {type(data)}")
    return SrfBlindVsAwareConfig(**data)


class _TinyIngestClassifier(nn.Module):
    """Minimal ingest + head model for SRF robustness checks."""

    def __init__(self, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.ingest = nn.ModuleDict(
            {
                "value_proj": nn.Linear(1, embed_dim),
                "wavelength_proj": nn.Linear(1, embed_dim),
                "width_proj": nn.Linear(1, embed_dim),
                "srf_proj": nn.Sequential(
                    nn.Linear(1, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim)
                ),
            }
        )
        self.quantity_embed = nn.Embedding(len(tuple(QuantityKind)), embed_dim)
        self.sensor_embed = nn.Embedding(1, embed_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.include_srf = True

    def forward(
        self,
        values: Tensor,
        wavelengths: Tensor,
        bandwidths: Tensor,
        quantity: Tensor,
        *,
        srf: Tensor | None,
    ) -> Tensor:
        if values.dim() == 1:
            values = values.unsqueeze(0)
        if wavelengths.dim() == 1:
            wavelengths = wavelengths.unsqueeze(0)
        if bandwidths.dim() == 1:
            bandwidths = bandwidths.unsqueeze(0)
        if quantity.dim() == 0:
            quantity = quantity.expand(values.shape[0])

        batch, bands = values.shape[0], values.shape[-1]
        vals = values.view(batch, bands, 1)
        wl = wavelengths.view(batch, bands, 1)
        bw = bandwidths.view(batch, bands, 1)

        emb = (
            self.ingest["value_proj"](vals)
            + self.ingest["wavelength_proj"](wl)
            + self.ingest["width_proj"](bw)
        )
        if srf is not None and self.include_srf:
            srf_vec = srf
            if srf_vec.dim() == 2:
                srf_vec = srf_vec.unsqueeze(0).expand(batch, -1, -1)
            srf_vec = srf_vec.view(batch, bands, -1).mean(dim=2, keepdim=True)
            emb = emb + self.ingest["srf_proj"](srf_vec)
        qty_emb = self.quantity_embed(quantity).view(batch, 1, -1).expand(-1, bands, -1)
        sensor_emb = self.sensor_embed.weight.view(1, 1, -1)
        pooled = (emb + qty_emb + sensor_emb).mean(dim=1)
        return self.head(pooled).squeeze(-1)


def _build_lab_samples(cfg: SrfBlindVsAwareConfig) -> tuple[list[Sample], torch.Tensor]:
    rng = np.random.default_rng(cfg.seed)
    axis = np.linspace(400.0, 1000.0, cfg.highres_points)

    samples: list[Sample] = []
    labels: list[float] = []
    for i in range(cfg.num_samples):
        base = 0.45 + 0.1 * np.sin(axis / 90.0 + 0.3 * i)
        narrow_feature = np.exp(-0.5 * ((axis - 720 - 5 * i) / 12.0) ** 2)
        class_offset = 0.04 if i % 2 else -0.04
        values = base + 0.15 * narrow_feature + class_offset
        values += rng.normal(0.0, 0.01, size=values.shape)
        label = float(i % 2)
        labels.append(label)
        samples.append(
            Sample(
                spectrum=Spectrum(wavelength_nm=axis, values=values, kind=QuantityKind.REFLECTANCE),
                sensor_id="lab",
            )
        )
    return samples, torch.tensor(labels, dtype=torch.float32)


def _project_dataset(cfg: SrfBlindVsAwareConfig) -> tuple[list[dict[str, Tensor | None]], torch.Tensor]:
    lab_samples, labels = _build_lab_samples(cfg)
    synth_cfg = SyntheticSensorConfig(
        highres_axis_nm=np.linspace(400.0, 1000.0, cfg.highres_points),
        n_bands=cfg.sensor_bands,
        center_jitter_nm=1.0,
        fwhm_range_nm=cfg.fwhm_range_nm,
        shape="gaussian",
        seed=cfg.seed,
    )
    dataset = SyntheticSensorDataset(lab_samples, synth_cfg)
    batched: list[dict[str, Tensor | None]] = []
    for item in dataset:
        srf_tensor = item["srf"]
        batched.append(
            {
                "values": item["tokens"],
                "wavelengths": item["wavelengths"],
                "bandwidths": item["bandwidths"],
                "srf": srf_tensor,
                "quantity": torch.tensor(int(QuantityKind.REFLECTANCE.value != QuantityKind.RADIANCE.value)),
            }
        )
    return batched, labels


def _evaluate(model: _TinyIngestClassifier, batch: dict[str, Tensor | None]) -> float:
    logits = model(
        batch["values"],
        batch["wavelengths"],
        batch["bandwidths"],
        batch["quantity"],
        srf=batch["srf"],
    )
    probs = torch.sigmoid(logits)
    return float((probs >= 0.5).float().mean())


def run_srf_blind_vs_aware_experiment(
    config: str | Path | Mapping[str, object] | None = None,
) -> dict[str, float]:
    """Train SRF-aware, then compare SRF-aware vs SRF-blind evaluation.

    Returns a report containing SRF-aware accuracy, SRF-blind accuracy, and the
    relative drop when disabling SRF embeddings.
    """

    cfg = load_srf_blind_vs_aware_config(config)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    batches, labels = _project_dataset(cfg)

    model = _TinyIngestClassifier(embed_dim=cfg.embed_dim, hidden_dim=cfg.hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(cfg.epochs):
        for batch, label in zip(batches, labels, strict=False):
            optimizer.zero_grad()
            logits = model(
                batch["values"],
                batch["wavelengths"],
                batch["bandwidths"],
                batch["quantity"],
                srf=batch["srf"],
            )
            loss = loss_fn(logits, label.unsqueeze(0))
            loss.backward()
            optimizer.step()

    # SRF-aware evaluation
    model.include_srf = True
    aware_scores = [_evaluate(model, batch) for batch in batches]
    aware_acc = float(np.mean(aware_scores))

    # SRF-blind evaluation: disable SRF embeddings but keep inputs identical.
    model.include_srf = False
    blind_scores = [_evaluate(model, batch) for batch in batches]
    blind_acc = float(np.mean(blind_scores))

    relative_drop = 0.0 if aware_acc == 0 else max(aware_acc - blind_acc, 0.0) / aware_acc

    report = {
        "aware_accuracy": aware_acc,
        "blind_accuracy": blind_acc,
        "relative_drop": relative_drop,
    }
    return report


__all__ = [
    "SrfBlindVsAwareConfig",
    "load_srf_blind_vs_aware_config",
    "run_srf_blind_vs_aware_experiment",
]
