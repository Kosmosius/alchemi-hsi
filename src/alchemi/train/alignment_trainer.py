"""CLIP-style alignment trainer used for Phase-2 experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import nn
import yaml

from ..align.batch_builders import NoiseConfig, build_emits_pairs
from ..align.cycle import CycleConfig, CycleReconstructionHeads
from ..align.losses import info_nce_symmetric
from ..eval.retrieval import retrieval_at_k, spectral_angle_deltas
from ..heads.banddepth import BandDepthHead, load_banddepth_config
from ..models.set_encoder import SetEncoder
from ..tokens.band_tokenizer import BandTokConfig, BandTokenizer
from ..tokens.registry import AxisUnit
from ..utils.logging import get_logger

_LOG = get_logger(__name__)


@dataclass(slots=True)
class TrainerConfig:
    batch_size: int = 8
    max_steps: int = 10
    log_every: int = 5
    eval_every: int = 5
    device: str = "cpu"
    dtype: str = "float32"
    seed: int = 0


@dataclass(slots=True)
class LabGridConfig:
    start: float = 380.0
    stop: float = 2500.0
    num: int = 256

    def to_array(self) -> np.ndarray:
        return np.linspace(self.start, self.stop, self.num, dtype=np.float64)


@dataclass(slots=True)
class DataConfig:
    sensor: str = "emit"
    lab_grid_nm: LabGridConfig = field(default_factory=LabGridConfig)
    lab_fwhm_nm: float = 5.0
    lab_noise_std: float = 0.01
    synthetic_peaks: int = 3
    sensor_noise_rel: float = 0.01


@dataclass(slots=True)
class ModelConfig:
    embed_dim: int = 64
    depth: int = 2
    heads: int = 4
    cycle_weight: float = 1.0


@dataclass(slots=True)
class OptimizerConfig:
    lr: float = 1e-3
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)


@dataclass(slots=True)
class LossConfig:
    tau_init: float = 0.07
    learnable_tau: bool = True
    gather_ddp: bool = True


@dataclass(slots=True)
class BandDepthConfig:
    enabled: bool = False
    weight: float = 0.1
    config: str | None = None
    hidden_dim: int | None = None
    loss: str = "l1"


@dataclass(slots=True)
class TokenizerConfig:
    axis_unit: AxisUnit = "nm"
    params: BandTokConfig = field(default_factory=BandTokConfig)


@dataclass(slots=True)
class AlignmentExperimentConfig:
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    cycle: CycleConfig = field(default_factory=CycleConfig)
    banddepth: BandDepthConfig = field(default_factory=BandDepthConfig)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "AlignmentExperimentConfig":
        def _build_trainer() -> TrainerConfig:
            cfg = data.get("trainer", {})
            return TrainerConfig(
                batch_size=int(cfg.get("batch_size", 8)),
                max_steps=int(cfg.get("max_steps", 10)),
                log_every=int(cfg.get("log_every", 5)),
                eval_every=int(cfg.get("eval_every", 5)),
                device=str(cfg.get("device", "cpu")),
                dtype=str(cfg.get("dtype", "float32")),
                seed=int(cfg.get("seed", 0)),
            )

        def _build_grid(raw: Mapping[str, Any] | Iterable[float] | None) -> LabGridConfig:
            if raw is None:
                return LabGridConfig()
            if isinstance(raw, Mapping):
                return LabGridConfig(
                    start=float(raw.get("start", 380.0)),
                    stop=float(raw.get("stop", 2500.0)),
                    num=int(raw.get("num", 256)),
                )
            values = np.asarray(tuple(raw), dtype=np.float64)
            if values.size < 2:
                raise ValueError("lab_grid_nm iterable must contain at least two samples")
            return LabGridConfig(start=float(values.min()), stop=float(values.max()), num=int(values.size))

        def _build_data() -> DataConfig:
            cfg = data.get("data", {})
            grid_raw = cfg.get("lab_grid_nm")
            grid_cfg = _build_grid(grid_raw)
            return DataConfig(
                sensor=str(cfg.get("sensor", "emit")),
                lab_grid_nm=grid_cfg,
                lab_fwhm_nm=float(cfg.get("lab_fwhm_nm", 5.0)),
                lab_noise_std=float(cfg.get("lab_noise_std", 0.01)),
                synthetic_peaks=int(cfg.get("synthetic_peaks", 3)),
                sensor_noise_rel=float(cfg.get("sensor_noise_rel", 0.01)),
            )

        def _build_tokenizer() -> TokenizerConfig:
            cfg = data.get("tokenizer", {})
            axis = cfg.get("axis_unit", "nm")
            params = {k: v for k, v in cfg.items() if k != "axis_unit"}
            return TokenizerConfig(axis_unit=axis, params=BandTokConfig(**params))

        def _build_model() -> ModelConfig:
            cfg = data.get("model", {})
            return ModelConfig(
                embed_dim=int(cfg.get("embed_dim", 64)),
                depth=int(cfg.get("depth", 2)),
                heads=int(cfg.get("heads", 4)),
                cycle_weight=float(cfg.get("cycle_weight", 1.0)),
            )

        def _build_optimizer() -> OptimizerConfig:
            cfg = data.get("optimizer", {})
            betas = tuple(cfg.get("betas", (0.9, 0.999)))
            return OptimizerConfig(
                lr=float(cfg.get("lr", 1e-3)),
                weight_decay=float(cfg.get("weight_decay", 0.0)),
                betas=(float(betas[0]), float(betas[1])),
            )

        def _build_loss() -> LossConfig:
            cfg = data.get("loss", {})
            return LossConfig(
                tau_init=float(cfg.get("tau_init", 0.07)),
                learnable_tau=bool(cfg.get("learnable_tau", True)),
                gather_ddp=bool(cfg.get("gather_ddp", True)),
            )

        def _build_cycle() -> CycleConfig:
            cfg = data.get("cycle", {})
            return CycleConfig(**cfg)

        def _build_banddepth() -> BandDepthConfig:
            cfg = data.get("banddepth", {})
            return BandDepthConfig(
                enabled=bool(cfg.get("enabled", False)),
                weight=float(cfg.get("weight", 0.1)),
                config=cfg.get("config"),
                hidden_dim=cfg.get("hidden_dim"),
                loss=str(cfg.get("loss", "l1")),
            )

        return cls(
            trainer=_build_trainer(),
            data=_build_data(),
            tokenizer=_build_tokenizer(),
            model=_build_model(),
            optimizer=_build_optimizer(),
            loss=_build_loss(),
            cycle=_build_cycle(),
            banddepth=_build_banddepth(),
        )


def load_alignment_config(path: str | Path) -> AlignmentExperimentConfig:
    """Load an alignment configuration from ``path``."""

    payload = yaml.safe_load(Path(path).read_text())
    if not isinstance(payload, Mapping):
        raise TypeError("Alignment config must be a mapping")
    return AlignmentExperimentConfig.from_mapping(payload)


class _TokenTower(nn.Module):
    """Shared transformer tower used by each modality."""

    def __init__(self, token_dim: int, embed_dim: int, depth: int, heads: int) -> None:
        super().__init__()
        self.token_proj = nn.Linear(token_dim, embed_dim)
        self.encoder = SetEncoder(dim=embed_dim, depth=depth, heads=heads)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        embedded = self.token_proj(tokens)
        pooled = self.encoder(embedded, mask)
        return self.norm(pooled)


@dataclass(slots=True)
class _TensorBatch:
    lab_tokens: torch.Tensor
    lab_mask: torch.Tensor
    sensor_tokens: torch.Tensor
    sensor_mask: torch.Tensor
    lab_values: torch.Tensor
    sensor_values: torch.Tensor
    sensor_wavelengths: torch.Tensor


class AlignmentTrainer:
    """Minimal CLIP-style trainer that couples lab and sensor embeddings."""

    def __init__(self, config: AlignmentExperimentConfig):
        self.cfg = config
        self._rng_lab = np.random.default_rng(self.cfg.trainer.seed)
        self._rng_sensor = np.random.default_rng(self.cfg.trainer.seed + 1)
        torch.manual_seed(self.cfg.trainer.seed)
        self.device = torch.device(self.cfg.trainer.device)
        self.dtype = _dtype_from_str(self.cfg.trainer.dtype)

        self.lab_wavelengths = self.cfg.data.lab_grid_nm.to_array()
        if self.lab_wavelengths.ndim != 1 or self.lab_wavelengths.size < 2:
            raise ValueError("Lab wavelength grid must be a 1-D array with >=2 samples")
        self.lab_fwhm = np.full_like(self.lab_wavelengths, self.cfg.data.lab_fwhm_nm)

        self.tokenizer = BandTokenizer(config=self.cfg.tokenizer.params)
        self.axis_unit = self.cfg.tokenizer.axis_unit
        self.token_dim = self._infer_token_dim()

        self.model = nn.ModuleDict(
            {
                "lab": _TokenTower(self.token_dim, self.cfg.model.embed_dim, self.cfg.model.depth, self.cfg.model.heads),
                "sensor": _TokenTower(self.token_dim, self.cfg.model.embed_dim, self.cfg.model.depth, self.cfg.model.heads),
            }
        ).to(self.device)

        self._tau_params: list[torch.nn.Parameter] = []
        if self.cfg.loss.learnable_tau:
            with torch.no_grad():
                dummy = torch.zeros(1, self.cfg.model.embed_dim, device=self.device, dtype=self.dtype)
                loss_out = info_nce_symmetric(
                    dummy,
                    dummy,
                    tau_init=self.cfg.loss.tau_init,
                    learnable_tau=True,
                    gather_ddp=self.cfg.loss.gather_ddp,
                )
            self._tau_params = loss_out.parameters()

        probe_pair = build_emits_pairs([(self.lab_wavelengths, np.ones_like(self.lab_wavelengths))], srf=self.cfg.data.sensor)[0]
        self.sensor_wavelengths = torch.as_tensor(
            probe_pair.sensor_wavelengths_nm, device=self.device, dtype=self.dtype
        )
        self.sensor_dim = int(self.sensor_wavelengths.numel())

        self.cycle_heads: CycleReconstructionHeads | None = None
        if self.cfg.cycle.enabled:
            self.cycle_heads = CycleReconstructionHeads(
                lab_dim=int(self.lab_wavelengths.size),
                sensor_dim=self.sensor_dim,
                config=self.cfg.cycle,
            ).to(device=self.device, dtype=self.dtype)

        self.banddepth_head: BandDepthHead | None = None
        if self.cfg.banddepth.enabled:
            if not self.cfg.banddepth.config:
                raise ValueError("banddepth.config must be provided when banddepth is enabled")
            bands = load_banddepth_config(self.cfg.banddepth.config)
            self.banddepth_head = BandDepthHead(
                embed_dim=self.cfg.model.embed_dim,
                bands=bands,
                hidden_dim=self.cfg.banddepth.hidden_dim,
                loss=self.cfg.banddepth.loss,
            ).to(device=self.device, dtype=self.dtype)

        self.noise_cfg = NoiseConfig(noise_level_rel=self.cfg.data.sensor_noise_rel, rng=self._rng_sensor)
        params: list[torch.nn.Parameter] = list(self.model.parameters())
        if self.cycle_heads is not None:
            params += list(self.cycle_heads.parameters())
        if self.banddepth_head is not None:
            params += list(self.banddepth_head.parameters())
        params += self._tau_params

        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.cfg.optimizer.lr,
            betas=self.cfg.optimizer.betas,
            weight_decay=self.cfg.optimizer.weight_decay,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AlignmentTrainer":
        return cls(load_alignment_config(path))

    def train(self, *, max_steps: int | None = None) -> list[dict[str, float]]:
        """Run the training loop and return loss history."""

        total_steps = int(max_steps or self.cfg.trainer.max_steps)
        history: list[dict[str, float]] = []
        log_every = max(1, self.cfg.trainer.log_every)
        eval_every = max(1, self.cfg.trainer.eval_every)
        for step in range(1, total_steps + 1):
            batch = self._build_batch(self.cfg.trainer.batch_size)
            metrics = self._train_step(batch)
            history.append({"step": float(step), **metrics})
            if step % log_every == 0:
                _LOG.info(
                    "step=%d loss=%.4f tau=%.4f", step, metrics["loss"], metrics.get("tau", 0.0)
                )
            if step % eval_every == 0:
                eval_metrics = self._evaluate(batch)
                _LOG.info(
                    "eval step=%d r@1=%.3f spectral_delta=%.4f",
                    step,
                    eval_metrics["retrieval@1"],
                    eval_metrics["spectral_delta"],
                )
        return history

    def _build_batch(self, batch_size: int) -> _TensorBatch:
        lab_spectra = self._sample_lab_spectra(batch_size)
        lab_batch = [(self.lab_wavelengths, spec) for spec in lab_spectra]
        pairs = build_emits_pairs(lab_batch, srf=self.cfg.data.sensor, noise_cfg=self.noise_cfg)
        lab_tokens, lab_masks, sensor_tokens, sensor_masks = [], [], [], []
        lab_values: list[np.ndarray] = []
        sensor_values: list[np.ndarray] = []
        for pair in pairs:
            lab_values.append(pair.lab_values)
            sensor_values.append(pair.sensor_values)
            lab_tok = self.tokenizer(
                pair.lab_values,
                pair.lab_wavelengths_nm,
                axis_unit=self.axis_unit,
                fwhm=self.lab_fwhm,
            )
            sensor_tok = self.tokenizer(
                pair.sensor_values,
                pair.sensor_wavelengths_nm,
                axis_unit=self.axis_unit,
            )
            lab_tokens.append(lab_tok.bands)
            sensor_tokens.append(sensor_tok.bands)
            lab_masks.append(~lab_tok.meta.invalid_mask)
            sensor_masks.append(~sensor_tok.meta.invalid_mask)

        lab_tokens_tensor = torch.as_tensor(np.stack(lab_tokens), device=self.device, dtype=self.dtype)
        lab_mask_tensor = torch.as_tensor(np.stack(lab_masks), device=self.device, dtype=torch.bool)
        sensor_tokens_tensor = torch.as_tensor(np.stack(sensor_tokens), device=self.device, dtype=self.dtype)
        sensor_mask_tensor = torch.as_tensor(np.stack(sensor_masks), device=self.device, dtype=torch.bool)
        lab_values_tensor = torch.as_tensor(np.stack(lab_values), device=self.device, dtype=self.dtype)
        sensor_values_tensor = torch.as_tensor(np.stack(sensor_values), device=self.device, dtype=self.dtype)
        sensor_wavelengths = torch.as_tensor(
            pairs[0].sensor_wavelengths_nm, device=self.device, dtype=self.dtype
        )
        return _TensorBatch(
            lab_tokens=lab_tokens_tensor,
            lab_mask=lab_mask_tensor,
            sensor_tokens=sensor_tokens_tensor,
            sensor_mask=sensor_mask_tensor,
            lab_values=lab_values_tensor,
            sensor_values=sensor_values_tensor,
            sensor_wavelengths=sensor_wavelengths,
        )

    def _encode_tokens(self, tokens: torch.Tensor, mask: torch.Tensor, tower: _TokenTower) -> torch.Tensor:
        embeddings: list[torch.Tensor] = []
        for idx in range(tokens.shape[0]):
            embeddings.append(tower(tokens[idx], mask[idx]))
        return torch.stack(embeddings, dim=0)

    def _train_step(self, batch: _TensorBatch) -> dict[str, float]:
        self.optimizer.zero_grad(set_to_none=True)
        z_lab = self._encode_tokens(batch.lab_tokens, batch.lab_mask, self.model["lab"])
        z_sensor = self._encode_tokens(batch.sensor_tokens, batch.sensor_mask, self.model["sensor"])
        loss_out = info_nce_symmetric(
            z_lab,
            z_sensor,
            tau_init=self.cfg.loss.tau_init,
            learnable_tau=self.cfg.loss.learnable_tau,
            gather_ddp=self.cfg.loss.gather_ddp,
        )
        total_loss = loss_out.loss
        metrics: dict[str, float] = {"loss": float(total_loss.detach().cpu())}
        if loss_out.metrics and "tau" in loss_out.metrics:
            metrics["tau"] = float(loss_out.metrics["tau"].detach().cpu())

        if self.cycle_heads is not None and self.cycle_heads.enabled:
            cycle_loss, breakdown = self.cycle_heads.cycle_loss(
                z_lab,
                z_sensor,
                batch.lab_values,
                {"radiance": batch.sensor_values},
            )
            total_loss = total_loss + self.cfg.model.cycle_weight * cycle_loss
            metrics["cycle_loss"] = float(cycle_loss.detach().cpu())
            for key, value in breakdown.items():
                metrics[f"cycle_{key}"] = float(value.detach().cpu())

        if self.banddepth_head is not None:
            preds = self.banddepth_head(z_sensor)
            targets = self.banddepth_head.compute_targets(batch.sensor_wavelengths, batch.sensor_values)
            band_loss = self.banddepth_head.loss(preds, targets)
            total_loss = total_loss + self.cfg.banddepth.weight * band_loss
            metrics["banddepth_loss"] = float(band_loss.detach().cpu())

        total_loss.backward()
        self.optimizer.step()
        metrics["loss"] = float(total_loss.detach().cpu())
        return metrics

    def _evaluate(self, batch: _TensorBatch) -> dict[str, float]:
        with torch.no_grad():
            z_lab = self._encode_tokens(batch.lab_tokens, batch.lab_mask, self.model["lab"])
            z_sensor = self._encode_tokens(batch.sensor_tokens, batch.sensor_mask, self.model["sensor"])
        z_lab_np = z_lab.detach().cpu().numpy()
        z_sensor_np = z_sensor.detach().cpu().numpy()
        gt = np.arange(z_lab_np.shape[0])
        retrieval = retrieval_at_k(z_lab_np, z_sensor_np, gt, k=1)
        delta = spectral_angle_deltas(z_lab_np, z_sensor_np)
        return {"retrieval@1": retrieval, "spectral_delta": delta["delta"]}

    def _sample_lab_spectra(self, batch: int) -> np.ndarray:
        lam = self.lab_wavelengths
        normed = (lam - lam.min()) / (lam.max() - lam.min())
        base = 0.5 + 0.5 * np.sin(2.0 * np.pi * normed)[None, :]
        spectra = np.repeat(base, batch, axis=0)
        for _ in range(self.cfg.data.synthetic_peaks):
            centers = self._rng_lab.uniform(lam.min(), lam.max(), size=(batch, 1))
            widths = self._rng_lab.uniform(20.0, 120.0, size=(batch, 1))
            amps = self._rng_lab.uniform(0.05, 0.3, size=(batch, 1))
            spectra += amps * np.exp(-0.5 * ((lam[None, :] - centers) / widths) ** 2)
        spectra += self._rng_lab.normal(scale=self.cfg.data.lab_noise_std, size=spectra.shape)
        return np.clip(spectra, 0.0, 2.0)

    def _infer_token_dim(self) -> int:
        dummy = self.tokenizer(
            np.ones_like(self.lab_wavelengths),
            self.lab_wavelengths,
            axis_unit=self.axis_unit,
            fwhm=self.lab_fwhm,
        )
        return int(dummy.bands.shape[-1])


def _dtype_from_str(name: str) -> torch.dtype:
    name = name.lower()
    if name in {"float32", "fp32"}:
        return torch.float32
    if name in {"float64", "fp64", "double"}:
        return torch.float64
    if name in {"float16", "fp16", "half"}:
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")
