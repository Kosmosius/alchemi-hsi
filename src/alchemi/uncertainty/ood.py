"""Out-of-distribution scoring utilities."""

from __future__ import annotations

from typing import Dict, Hashable, Iterable, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def softmax_complement_score(logits: torch.Tensor) -> torch.Tensor:
    """1 - max softmax probability as an OOD score."""

    probs = F.softmax(logits, dim=-1)
    max_prob, _ = probs.max(dim=-1)
    return 1.0 - max_prob


def energy_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Energy-based OOD score (more negative implies OOD)."""

    scaled = logits / temperature
    return -temperature * torch.logsumexp(scaled, dim=-1)


def mahalanobis_distance(
    embedding: torch.Tensor,
    class_means: torch.Tensor,
    covariance_inv: torch.Tensor,
) -> torch.Tensor:
    """Compute Mahalanobis distance from an embedding to each class mean."""

    diff = embedding.unsqueeze(0) - class_means
    left = torch.matmul(diff, covariance_inv)
    return torch.sum(left * diff, dim=-1)


def mahalanobis_ood_score(
    embedding: torch.Tensor,
    class_means: torch.Tensor,
    covariance_inv: torch.Tensor,
) -> torch.Tensor:
    """OOD score based on minimum Mahalanobis distance to known classes."""

    distances = mahalanobis_distance(embedding, class_means, covariance_inv)
    return distances.min(dim=0).values


def spectral_angle_mapper(
    spectrum: torch.Tensor,
    prototypes: torch.Tensor,
) -> torch.Tensor:
    """Compute the smallest spectral angle between spectrum and prototypes."""

    spectrum_norm = F.normalize(spectrum, dim=-1)
    proto_norm = F.normalize(prototypes, dim=-1)
    cosine = torch.matmul(proto_norm, spectrum_norm.unsqueeze(-1)).squeeze(-1)
    angles = torch.arccos(torch.clamp(cosine, -1 + 1e-7, 1 - 1e-7))
    return angles.min(dim=0).values


def combine_ood_scores(
    scores: Mapping[str, torch.Tensor],
    *,
    weights: Optional[Mapping[str, float]] = None,
    aggregation: str = "mean",
    threshold: Optional[float] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Combine multiple OOD components into a scalar score or decision.

    Args:
        scores: Mapping from score name to tensor of shape ``(N,)``.
        weights: Optional weights for each score. Defaults to uniform.
        aggregation: ``"mean"`` or ``"max"``.
        threshold: Optional decision threshold; if provided, returns a boolean
            tensor indicating OOD detections.
    """

    if weights is None:
        weights = {name: 1.0 for name in scores}

    stacked = []
    for name, score in scores.items():
        weight = weights.get(name, 1.0)
        stacked.append(weight * score)

    aggregated = torch.stack(stacked, dim=0)
    if aggregation == "mean":
        combined = aggregated.mean(dim=0)
    elif aggregation == "max":
        combined = aggregated.max(dim=0).values
    else:
        raise ValueError("aggregation must be 'mean' or 'max'")

    decision = None
    if threshold is not None:
        decision = combined > threshold

    return combined, decision
