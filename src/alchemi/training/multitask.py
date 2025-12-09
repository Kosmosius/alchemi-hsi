"""Multi-task balancing utilities (static weights, GradNorm, PCGrad)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch

from alchemi.config.core import MultiTaskConfig


@dataclass
class MultiTaskLoss:
    cfg: MultiTaskConfig
    weights: Dict[str, float] = field(default_factory=dict)
    _initialised: bool = False

    def _maybe_init(self, losses: dict[str, torch.Tensor]) -> None:
        if self._initialised:
            return
        self.weights = {
            "solids": self.cfg.solids_weight,
            "gas": self.cfg.gas_weight,
            "aux": self.cfg.aux_weight,
        }
        for name in losses:
            self.weights.setdefault(name, 1.0)
        self._initialised = True

    def combine(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        self._maybe_init(losses)
        weighted = []
        for name, loss in losses.items():
            weight = self.weights.get(name, 1.0)
            weighted.append(weight * loss)
        total = (
            sum(weighted)
            if weighted
            else torch.zeros((), device=next(iter(losses.values())).device)
        )
        if self.cfg.use_gradnorm:
            self._update_gradnorm(losses)
        return total

    def _update_gradnorm(self, losses: dict[str, torch.Tensor], alpha: float = 0.5) -> None:
        # Simple GradNorm approximation based on relative loss magnitudes.
        magnitudes = {name: loss.detach().abs().mean().item() for name, loss in losses.items()}
        mean_mag = sum(magnitudes.values()) / max(len(magnitudes), 1)
        for name, mag in magnitudes.items():
            target = (mag / (mean_mag + 1e-6)) ** alpha
            self.weights[name] = float(0.9 * self.weights.get(name, 1.0) + 0.1 * target)

    def pcgrad(self, params: list[torch.Tensor]) -> None:
        if not self.cfg.use_pcgrad:
            return
        # Lightweight PCGrad: normalise gradients and project conflicting components.
        grads = [p.grad.view(-1) for p in params if p.grad is not None]
        for i, gi in enumerate(grads):
            for j, gj in enumerate(grads):
                if i >= j:
                    continue
                dot = torch.dot(gi, gj)
                if dot < 0:
                    proj = dot / (gj.norm() ** 2 + 1e-6)
                    gi -= proj * gj
        # Write back
        idx = 0
        for p in params:
            if p.grad is None:
                continue
            p.grad.copy_(grads[idx].view_as(p.grad))
            idx += 1


__all__ = ["MultiTaskLoss"]
