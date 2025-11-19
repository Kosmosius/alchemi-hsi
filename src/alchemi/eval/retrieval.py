from __future__ import annotations
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class RetrievalMetrics:
    recall: float
    precision: float


def compute_retrieval_at_k(
    q: np.ndarray, keys: np.ndarray, k: int = 1
) -> RetrievalMetrics:
    """Return recall@k and precision@k assuming 1:1 lab↔sensor ordering."""

    if q.ndim != 2 or keys.ndim != 2:
        raise ValueError("Embeddings must be 2-D matrices")
    if q.shape[0] != keys.shape[0]:
        raise ValueError("Query and database sizes must match for paired retrieval")
    k = max(1, int(k))
    indices = _topk_indices(q, keys, k)
    gt = np.arange(q.shape[0])
    hits = (indices == gt[:, None]).any(axis=1)
    recall = float(hits.mean())
    precision = float(hits.sum() / (k * max(1, q.shape[0])))
    return RetrievalMetrics(recall=recall, precision=precision)


def retrieval_at_k(q: np.ndarray, keys: np.ndarray, gt: np.ndarray, k: int = 1) -> float:
    """Compute recall@k for L2-normalised embeddings."""

    if q.ndim != 2 or keys.ndim != 2:
        raise ValueError("Embeddings must be 2-D matrices")
    if q.shape[0] != gt.shape[0]:
        raise ValueError("Ground-truth indices must align with queries")
    k = max(1, int(k))
    indices = _topk_indices(q, keys, k)
    hits = (indices == gt[:, None]).any(axis=1).mean()
    return float(hits)


def retrieval_summary(
    q: np.ndarray,
    keys: np.ndarray,
    gt: np.ndarray,
    ks: tuple[int, ...] = (1, 5),
) -> dict[str, float]:
    """Return a dictionary of recall@k metrics for the provided ``ks``."""

    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"retrieval@{k}"] = retrieval_at_k(q, keys, gt, k)
    return metrics


def spectral_angle_deltas(z_lab: np.ndarray, z_sensor: np.ndarray) -> dict[str, float]:
    """Return mean spectral angle for matched vs mismatched pairs."""

    matched = _spectral_angle(z_lab, z_sensor)
    if z_lab.shape[0] > 1:
        rolled = np.roll(z_sensor, shift=1, axis=0)
        mismatched = _spectral_angle(z_lab, rolled)
    else:
        mismatched = matched
    mean_matched = float(np.mean(matched))
    mean_mismatched = float(np.mean(mismatched))
    return {
        "matched": mean_matched,
        "mismatched": mean_mismatched,
        "delta": mean_mismatched - mean_matched,
    }


def _normalize(arr: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr / denom


def _spectral_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape != b.shape:
        raise ValueError("Spectral angle inputs must share shape")
    dot = np.sum(_normalize(a) * _normalize(b), axis=1)
    dot = np.clip(dot, -1.0, 1.0)
    return np.arccos(dot)


def _topk_indices(q: np.ndarray, keys: np.ndarray, k: int) -> np.ndarray:
    qn = _normalize(q)
    kn = _normalize(keys)
    sims = qn @ kn.T
    return np.argsort(-sims, axis=1)[:, :k]


def random_retrieval_at_k(num_queries: int, num_db: int, k: int = 1) -> float:
    """Return the expected retrieval@k score for random guessing."""

    if num_db <= 0:
        raise ValueError("num_db must be positive")
    k = max(1, int(k))
    return float(min(k, num_db) / num_db)
