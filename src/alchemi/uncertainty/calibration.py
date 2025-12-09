"""Calibration utilities for uncertainty estimation."""

from __future__ import annotations

from itertools import pairwise
from typing import Dict, Hashable, Iterable, Mapping, Optional

import torch
import torch.nn.functional as F

TemperatureLike = float | torch.Tensor | Mapping[Hashable, float]


def _make_temperature_vector(
    temperature: TemperatureLike,
    num_samples: int,
    device: torch.device,
    ids: Optional[Iterable[Hashable]] = None,
) -> torch.Tensor:
    """Create a temperature tensor for each sample.

    Args:
        temperature: Scalar, tensor, or mapping from ids to temperature values.
        num_samples: Number of samples to create temperatures for.
        device: Target device.
        ids: Optional iterable of ids matching the samples.

    Returns:
        Tensor of shape (num_samples, 1) containing temperatures.
    """

    if isinstance(temperature, Mapping):
        if ids is None:
            raise ValueError("ids must be provided when using a temperature mapping")
        temps = []
        for key in ids:
            if key not in temperature:
                raise KeyError(f"Temperature for key {key!r} is missing")
            temps.append(float(temperature[key]))
        temp_tensor = torch.tensor(temps, device=device, dtype=torch.float32).unsqueeze(1)
    else:
        temp_tensor = torch.as_tensor(temperature, device=device, dtype=torch.float32)
        if temp_tensor.ndim == 0:
            temp_tensor = temp_tensor.repeat(num_samples).unsqueeze(1)
        elif temp_tensor.ndim == 1:
            if temp_tensor.numel() not in (1, num_samples):
                raise ValueError("Temperature vector must be length 1 or match batch size")
            if temp_tensor.numel() == 1:
                temp_tensor = temp_tensor.repeat(num_samples).unsqueeze(1)
            else:
                temp_tensor = temp_tensor.unsqueeze(1)
        else:
            raise ValueError("Temperature must be scalar or 1D tensor")

    return torch.clamp(temp_tensor, min=1e-6)


def temperature_scale_logits(
    logits: torch.Tensor,
    temperature: TemperatureLike,
    *,
    sensor_ids: Optional[Iterable[Hashable]] = None,
    task_ids: Optional[Iterable[Hashable]] = None,
) -> torch.Tensor:
    """Scale logits by temperature.

    A higher temperature produces softer probability distributions. Temperatures
    can be provided per sensor or per task via a mapping.

    Args:
        logits: Raw, unnormalized model outputs of shape ``(N, C)``.
        temperature: Scalar, tensor, or mapping from ids to temperatures.
        sensor_ids: Optional sensor identifiers aligned with ``logits``.
        task_ids: Optional task identifiers aligned with ``logits``.

    Returns:
        Logits scaled by the appropriate temperatures.
    """

    if sensor_ids is not None and task_ids is not None:
        raise ValueError("Provide at most one of sensor_ids or task_ids")

    ids = sensor_ids if sensor_ids is not None else task_ids
    temp_vector = _make_temperature_vector(temperature, logits.shape[0], logits.device, ids)
    return logits / temp_vector


def _fit_group_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    max_iters: int = 200,
    lr: float = 0.05,
) -> float:
    """Fit a single temperature parameter by minimizing NLL."""

    log_t = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.Adam([log_t], lr=lr)

    for _ in range(max_iters):
        optimizer.zero_grad()
        temperature = F.softplus(log_t) + 1e-6
        scaled_logits = logits / temperature
        loss = F.cross_entropy(scaled_logits, labels)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        return float(F.softplus(log_t).item())


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    sensor_ids: Optional[Iterable[Hashable]] = None,
    task_ids: Optional[Iterable[Hashable]] = None,
    max_iters: int = 200,
    lr: float = 0.05,
) -> float | Dict[Hashable, float]:
    """Fit temperature scaling parameters on validation data.

    Args:
        logits: Validation logits ``(N, C)``.
        labels: Ground-truth labels ``(N,)``.
        sensor_ids: Optional iterable of sensor identifiers per sample.
        task_ids: Optional iterable of task identifiers per sample.
        max_iters: Maximum optimization iterations for each group.
        lr: Learning rate for the optimizer.

    Returns:
        A scalar temperature if no grouping is provided, otherwise a mapping
        from group identifier to temperature.
    """

    if sensor_ids is not None and task_ids is not None:
        raise ValueError("Provide at most one of sensor_ids or task_ids")

    ids = sensor_ids if sensor_ids is not None else task_ids
    if ids is None:
        return _fit_group_temperature(logits, labels, max_iters=max_iters, lr=lr)

    id_tensor = torch.as_tensor(list(ids), device=logits.device)
    unique_ids = torch.unique(id_tensor)
    temperatures: Dict[Hashable, float] = {}

    for uid in unique_ids:
        mask = id_tensor == uid
        group_logits = logits[mask]
        group_labels = labels[mask]
        temperatures[int(uid.item()) if uid.numel() == 1 else uid.item()] = _fit_group_temperature(
            group_logits, group_labels, max_iters=max_iters, lr=lr
        )

    return temperatures


def expected_calibration_error(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> torch.Tensor:
    """Compute the Expected Calibration Error (ECE).

    Args:
        logits: Logits ``(N, C)``.
        labels: Ground-truth labels ``(N,)``.
        n_bins: Number of bins for confidence partitioning.

    Returns:
        Scalar tensor with the ECE value.
    """

    probs = logits.softmax(dim=-1)
    confidences, predictions = probs.max(dim=-1)
    accuracies = predictions.eq(labels)

    bins = torch.linspace(0, 1, steps=n_bins + 1, device=logits.device)
    ece = torch.zeros((), device=logits.device)

    for bin_lower, bin_upper in pairwise(bins):
        mask = (confidences > bin_lower) & (confidences <= bin_upper)
        if mask.any():
            bin_acc = accuracies[mask].float().mean()
            bin_conf = confidences[mask].mean()
            ece += (mask.float().mean()) * torch.abs(bin_acc - bin_conf)

    return ece


def negative_log_likelihood(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute mean negative log-likelihood (cross entropy)."""

    return F.cross_entropy(logits, labels)


def brier_score(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute the Brier score for multi-class predictions."""

    probs = logits.softmax(dim=-1)
    one_hot = F.one_hot(labels, num_classes=probs.shape[-1]).float()
    return torch.mean(torch.sum((probs - one_hot) ** 2, dim=-1))


def calibration_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> Dict[str, torch.Tensor]:
    """Compute standard calibration metrics for a batch."""

    return {
        "ece": expected_calibration_error(logits, labels, n_bins=n_bins),
        "nll": negative_log_likelihood(logits, labels),
        "brier": brier_score(logits, labels),
    }
