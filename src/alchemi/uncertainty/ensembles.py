"""Lightweight ensemble wrappers for uncertainty estimation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence

import torch
from torch import nn


@dataclass
class EnsembleOutput:
    """Output container for ensemble predictions."""

    mean: torch.Tensor
    variance: torch.Tensor
    member_outputs: torch.Tensor


class TorchEnsemble:
    """Simple manager for a collection of PyTorch models or heads."""

    def __init__(self, members: Sequence[nn.Module]):
        if len(members) == 0:
            raise ValueError("Ensemble must contain at least one member")
        self.members: List[nn.Module] = list(members)

    def to(self, device: torch.device) -> "TorchEnsemble":
        for member in self.members:
            member.to(device)
        return self

    def eval(self) -> "TorchEnsemble":
        for member in self.members:
            member.eval()
        return self

    def train(self) -> "TorchEnsemble":
        for member in self.members:
            member.train()
        return self

    def forward_member(self, member: nn.Module, *args, **kwargs) -> torch.Tensor:
        return member(*args, **kwargs)

    def predict(self, *args, **kwargs) -> EnsembleOutput:
        """Run all ensemble members and aggregate outputs.

        Returns:
            EnsembleOutput with mean, variance, and stacked member outputs.
        """

        outputs = []
        with torch.no_grad():
            for member in self.members:
                outputs.append(self.forward_member(member, *args, **kwargs))
        member_outputs = torch.stack(outputs, dim=0)
        mean = member_outputs.mean(dim=0)
        variance = member_outputs.var(dim=0, unbiased=False)
        return EnsembleOutput(mean=mean, variance=variance, member_outputs=member_outputs)


def train_ensemble(
    members: Sequence[nn.Module],
    train_step: Callable[[nn.Module], None],
) -> TorchEnsemble:
    """Train each ensemble member using the provided callable.

    The callable should encapsulate a full training loop (optimizer steps,
    dataloaders, etc.) for an individual model. This keeps training logic
    framework-specific while preserving a uniform interface.
    """

    for member in members:
        train_step(member)
    return TorchEnsemble(members)


def run_ensemble(
    ensemble: TorchEnsemble,
    inputs: Iterable,
    forward_kwargs: dict | None = None,
) -> EnsembleOutput:
    """Convenience wrapper to run an ensemble over an iterable of inputs."""

    forward_kwargs = forward_kwargs or {}
    outputs = []
    with torch.no_grad():
        for batch in inputs:
            outputs.append(ensemble.predict(batch, **forward_kwargs))

    member_outputs = torch.stack([o.member_outputs for o in outputs], dim=1)
    mean = member_outputs.mean(dim=(0, 1))
    variance = member_outputs.var(dim=(0, 1), unbiased=False)
    return EnsembleOutput(mean=mean, variance=variance, member_outputs=member_outputs)
