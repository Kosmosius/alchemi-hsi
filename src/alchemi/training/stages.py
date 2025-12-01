"""Stage-specific training entrypoints (A/B/C/D)."""
from __future__ import annotations

from alchemi.config.core import TrainingConfig
from alchemi.training.loops import evaluate, train_epoch


def run_mae_pretrain(trainer, train_loader, val_loader, cfg: TrainingConfig) -> None:
    epochs = cfg.stages.mae.epochs or 1
    for epoch in range(epochs):
        train_epoch(
            train_loader,
            trainer.mae_step,
            trainer.optimizer,
            scheduler=trainer.scheduler,
            accumulation_steps=cfg.accumulation_steps,
            logger=trainer.logger,
        )
        if epoch % max(1, epochs // 2) == 0:
            evaluate(val_loader, trainer.eval_step, logger=trainer.logger)


def run_alignment(trainer, train_loader, val_loader, cfg: TrainingConfig) -> None:
    epochs = cfg.stages.align.epochs or 1
    for epoch in range(epochs):
        train_epoch(
            train_loader,
            trainer.alignment_step,
            trainer.optimizer,
            scheduler=trainer.scheduler,
            accumulation_steps=cfg.accumulation_steps,
            logger=trainer.logger,
        )
        if epoch % max(1, epochs // 2) == 0:
            evaluate(val_loader, trainer.eval_step, logger=trainer.logger)


def run_task_training(trainer, train_loader, val_loader, cfg: TrainingConfig) -> None:
    epochs = cfg.stages.tasks.epochs or 1
    for epoch in range(epochs):
        train_epoch(
            train_loader,
            trainer.task_step,
            trainer.optimizer,
            scheduler=trainer.scheduler,
            accumulation_steps=cfg.accumulation_steps,
            logger=trainer.logger,
        )
        if epoch % max(1, epochs // 2) == 0:
            evaluate(val_loader, trainer.eval_step, logger=trainer.logger)


def run_uncertainty_calibration(trainer, train_loader, val_loader, cfg: TrainingConfig) -> None:
    epochs = cfg.stages.uncertainty.epochs or 1
    for epoch in range(epochs):
        train_epoch(
            train_loader,
            trainer.task_step,
            trainer.optimizer,
            scheduler=trainer.scheduler,
            accumulation_steps=cfg.accumulation_steps,
            logger=trainer.logger,
        )
        if epoch % max(1, epochs // 2) == 0:
            evaluate(val_loader, trainer.eval_step, logger=trainer.logger)


__all__ = [
    "run_mae_pretrain",
    "run_alignment",
    "run_task_training",
    "run_uncertainty_calibration",
]
