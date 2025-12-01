"""High level training orchestrator for staged experiments.

The trainer glues together configuration objects, data pipelines, model
construction, loss wiring, optimisation, and logging. The implementation here
is intentionally lightweight – a dummy call such as ``Trainer(cfg).run()`` will
build the model stack and iterate through a few batches without requiring an
external dataset. Detailed loss wiring can be extended later without changing
this entry point.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from alchemi.config import ExperimentConfig, StageSchedule
from alchemi.data import pipelines
from alchemi.models import (
    AnySensorIngest,
    AuxHead,
    GasHead,
    LabOverheadAlignment,
    MAEBackbone,
    SolidsHead,
)

from .losses import (
    build_aux_loss,
    build_gas_loss,
    build_info_nce_loss,
    build_mae_reconstruction_loss,
    build_solids_loss,
)
from .loops import evaluate, train_epoch
from .logging import MetricLogger
from .multitask import MultiTaskLoss
from .optimizers import build_optimizer
from .schedulers import build_scheduler
from .stages import (
    run_alignment,
    run_mae_pretrain,
    run_task_training,
    run_uncertainty_calibration,
)


@dataclass
class _DummySpectra(Dataset[dict[str, torch.Tensor]]):
    """Fallback dataset emitting random spectral tokens.

    This keeps the trainer runnable in environments without data availability
    while still exercising the model and loss plumbing.
    """

    length: int
    embed_dim: int
    seq_len: int = 128

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # pragma: no cover - simple
        tokens = torch.randn(self.seq_len, self.embed_dim)
        alt_tokens = torch.randn(self.seq_len, self.embed_dim)
        labels = torch.randint(0, 5, (1,))
        return {"tokens": tokens, "tokens_alt": alt_tokens, "labels": labels.float()}


class Trainer:
    """Top-level staged training orchestrator."""

    def __init__(self, cfg: ExperimentConfig, *, device: torch.device | None = None) -> None:
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = MetricLogger(enable_wandb=False)

        # Build model components from config
        self.ingest = AnySensorIngest.from_config(cfg.model).to(self.device)
        self.backbone = MAEBackbone.from_config(cfg.model).to(self.device)
        self.alignment = LabOverheadAlignment.from_config(cfg.model).to(self.device)
        self.solids_head = SolidsHead.from_config(cfg.model.backbone.embed_dim, cfg.model.heads.solids).to(
            self.device
        )
        self.gas_head = GasHead.from_config(cfg.model.backbone.embed_dim, cfg.model.heads.gas).to(self.device)
        self.aux_head = AuxHead.from_config(cfg.model.backbone.embed_dim, cfg.model.heads.aux).to(self.device)

        # Loss helpers
        self.mae_loss = build_mae_reconstruction_loss()
        self.info_nce = build_info_nce_loss(cfg.model.alignment.temperature)
        self.solids_loss = build_solids_loss()
        self.gas_loss = build_gas_loss()
        self.aux_loss = build_aux_loss()

        self.multitask = MultiTaskLoss(cfg.training.multitask)

        # Optimizer + scheduler will be constructed per-stage in case LR differs
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler._LRScheduler | None = None

        self.train_loader, self.val_loader = self._build_pipelines()

    def _build_pipelines(self) -> tuple[DataLoader[Any], DataLoader[Any]]:
        """Build data pipelines; fall back to dummy random data when unavailable."""

        batch_size = self.cfg.training.batch_size
        try:
            loaders: list[DataLoader[Any]] = []
            for name in self.cfg.data.dataset_names:
                builder = getattr(pipelines, f"build_{name}_pipeline", None)
                if builder is None:
                    continue
                loaders.append(builder(self.cfg.data))
            if loaders:
                return loaders[0], loaders[0]
        except Exception:
            # Fallback below
            pass

        dummy = _DummySpectra(length=32, embed_dim=self.cfg.model.backbone.embed_dim)
        loader = DataLoader(dummy, batch_size=batch_size)
        return loader, loader

    def _set_stage_optim(self, schedule: StageSchedule, stage_name: str) -> None:
        stage_cfg = getattr(schedule, stage_name)
        self.optimizer = build_optimizer(
            self.parameters(),
            self.cfg.training.optimizer,
            lr_override=stage_cfg.learning_rate,
        )
        self.scheduler = build_scheduler(self.optimizer, self.cfg.training.scheduler, stage_cfg.epochs)

    def parameters(self) -> Iterable[nn.Parameter]:  # pragma: no cover - passthrough
        modules = [self.ingest, self.backbone, self.alignment, self.solids_head, self.gas_head, self.aux_head]
        for module in modules:
            yield from module.parameters()

    def run(self) -> None:
        """Run all configured training stages sequentially."""

        stages = self.cfg.training.stages
        if stages.mae.enabled:
            self._set_stage_optim(stages, "mae")
            run_mae_pretrain(self, self.train_loader, self.val_loader, self.cfg.training)

        if stages.align.enabled:
            self._set_stage_optim(stages, "align")
            run_alignment(self, self.train_loader, self.val_loader, self.cfg.training)

        if stages.tasks.enabled:
            self._set_stage_optim(stages, "tasks")
            run_task_training(self, self.train_loader, self.val_loader, self.cfg.training)

        if stages.uncertainty.enabled:
            self._set_stage_optim(stages, "uncertainty")
            run_uncertainty_calibration(self, self.train_loader, self.val_loader, self.cfg.training)

    # ------------------------------------------------------------------
    # Convenience wrappers used by stages
    # ------------------------------------------------------------------
    def mae_step(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        tokens = batch["tokens"].to(self.device)
        out = self.backbone.forward_mae(tokens)
        loss = self.mae_loss(out.decoded, tokens, mask=out.mask)
        return loss, {"mae": float(loss.detach().cpu())}

    def alignment_step(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        tok_a = batch["tokens"].to(self.device)
        tok_b = batch.get("tokens_alt", tok_a).to(self.device)
        enc_a = self.backbone.forward_encoder(tok_a)
        enc_b = self.backbone.forward_encoder(tok_b)
        loss = self.info_nce(enc_a.mean(dim=1), enc_b.mean(dim=1))
        return loss, {"info_nce": float(loss.detach().cpu())}

    def task_step(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        tokens = batch["tokens"].to(self.device)
        features = self.backbone.forward_encoder(tokens)
        pooled = features.mean(dim=1)
        spatial = pooled.view(pooled.size(0), 1, 1, -1)
        losses: dict[str, torch.Tensor] = {}

        solids_out = self.solids_head(pooled)
        losses["solids"] = self.solids_loss(solids_out, batch.get("labels"))

        gas_out = self.gas_head(spatial)
        losses["gas"] = self.gas_loss(gas_out, batch.get("gas_labels"))

        aux_out = self.aux_head(spatial)
        losses["aux"] = self.aux_loss(aux_out, batch.get("aux_targets"))

        total = self.multitask.combine(losses)
        metrics = {k: float(v.detach().cpu()) for k, v in losses.items()}
        return total, metrics

    def eval_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        loss, metrics = self.task_step(batch)
        metrics = {**metrics, "total": float(loss.detach().cpu())}
        return metrics


def run_pretrain_mae(config: ExperimentConfig) -> None:
    """Compatibility wrapper around the new :class:`Trainer` API."""

    trainer = Trainer(config)
    trainer.cfg.training.stages.align.enabled = False
    trainer.cfg.training.stages.tasks.enabled = False
    trainer.cfg.training.stages.uncertainty.enabled = False
    trainer.run()


def run_eval(config: ExperimentConfig) -> None:
    """Run a quick evaluation loop over the validation loader."""

    trainer = Trainer(config)
    evaluate(trainer.val_loader, trainer.eval_step, logger=trainer.logger)


__all__ = ["Trainer", "run_pretrain_mae", "run_eval"]
