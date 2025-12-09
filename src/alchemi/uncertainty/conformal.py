"""Split-conformal prediction helpers for classification and regression."""

from __future__ import annotations

from typing import Dict, Hashable, Iterable, Mapping, Optional, Tuple

import torch

GroupThresholds = Mapping[Hashable, float]


def _compute_quantiles(
    scores: torch.Tensor,
    group_ids: Optional[Iterable[Hashable]],
    alpha: float,
) -> Dict[Hashable, float]:
    id_tensor: Optional[torch.Tensor]
    if group_ids is not None:
        id_tensor = torch.as_tensor(list(group_ids), device=scores.device)
    else:
        id_tensor = None

    thresholds: Dict[Hashable, float] = {}
    if id_tensor is None:
        q = torch.quantile(scores, 1 - alpha, interpolation="higher")
        thresholds["global"] = float(q.item())
        return thresholds

    unique_ids = torch.unique(id_tensor)
    for uid in unique_ids:
        mask = id_tensor == uid
        q = torch.quantile(scores[mask], 1 - alpha, interpolation="higher")
        thresholds[int(uid.item()) if uid.numel() == 1 else uid.item()] = float(q.item())
    return thresholds


# Classification


def classification_conformal_thresholds(
    calib_probs: torch.Tensor,
    calib_labels: torch.Tensor,
    *,
    group_ids: Optional[Iterable[Hashable]] = None,
    alpha: float = 0.1,
) -> GroupThresholds:
    """Compute nonconformity score thresholds for classification."""

    true_probs = calib_probs[torch.arange(calib_probs.size(0)), calib_labels]
    scores = 1.0 - true_probs
    return _compute_quantiles(scores, group_ids, alpha)


def classification_label_set(
    probs: torch.Tensor,
    thresholds: GroupThresholds,
    *,
    group_id: Optional[Hashable] = None,
) -> torch.Tensor:
    """Return the conformal prediction label set for a sample."""

    threshold = thresholds.get(group_id, thresholds.get("global"))
    if threshold is None:
        raise KeyError("No threshold found for provided group id")
    return torch.nonzero(probs >= (1.0 - threshold), as_tuple=False).flatten()


# Regression


def regression_conformal_thresholds(
    calib_predictions: torch.Tensor,
    calib_targets: torch.Tensor,
    *,
    group_ids: Optional[Iterable[Hashable]] = None,
    alpha: float = 0.1,
) -> GroupThresholds:
    """Compute residual quantiles for regression conformal intervals."""

    residuals = torch.abs(calib_predictions - calib_targets)
    return _compute_quantiles(residuals, group_ids, alpha)


def regression_interval(
    prediction: torch.Tensor,
    thresholds: GroupThresholds,
    *,
    group_id: Optional[Hashable] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return a predictive interval centered on the prediction."""

    threshold = thresholds.get(group_id, thresholds.get("global"))
    if threshold is None:
        raise KeyError("No threshold found for provided group id")
    lower = prediction - threshold
    upper = prediction + threshold
    return lower, upper
