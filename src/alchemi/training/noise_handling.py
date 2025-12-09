"""Label noise handling helpers."""

from __future__ import annotations

from typing import Iterable

import torch

Tensor = torch.Tensor


def filter_by_confidence(confidence: Tensor, threshold: float = 0.5) -> Tensor:
    """Return weights down-weighting samples below a confidence threshold."""

    return torch.where(
        confidence >= threshold, torch.ones_like(confidence), torch.zeros_like(confidence)
    )


def mutual_refinement(predictions: Iterable[Tensor], momentum: float = 0.5) -> Tensor:
    """Simple self-refinement averaging ensemble predictions."""

    preds = list(predictions)
    if not preds:
        raise ValueError("No predictions provided for refinement")
    avg = torch.stack(preds).mean(dim=0)
    refined = []
    for pred in preds:
        refined.append(momentum * pred + (1 - momentum) * avg)
    return torch.stack(refined).mean(dim=0)


def soft_labels_from_teacher(teacher_logits: Tensor, temperature: float = 1.0) -> Tensor:
    """Convert teacher logits to smoothed labels."""

    return torch.softmax(teacher_logits / max(temperature, 1e-4), dim=-1)


__all__ = ["filter_by_confidence", "mutual_refinement", "soft_labels_from_teacher"]
