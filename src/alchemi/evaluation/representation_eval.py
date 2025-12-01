"""Representation-level evaluation utilities."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .metrics import accuracy


def recall_at_k(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    query_labels: Sequence[int],
    gallery_labels: Sequence[int],
    k: int = 1,
) -> float:
    """Compute Recall@K for retrieval tasks."""

    query_norm = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-12)
    gallery_norm = gallery_embeddings / (np.linalg.norm(gallery_embeddings, axis=1, keepdims=True) + 1e-12)

    sims = query_norm @ gallery_norm.T
    correct = 0
    for i, label in enumerate(query_labels):
        top_k = np.argpartition(-sims[i], k - 1)[:k]
        if np.any(np.asarray(gallery_labels)[top_k] == label):
            correct += 1
    return float(correct / len(query_labels))


def linear_probe_cross_sensor(
    train_features: np.ndarray,
    train_labels: Sequence[int],
    test_features: np.ndarray,
    test_labels: Sequence[int],
) -> Mapping[str, float]:
    """Train a simple linear probe on one sensor and evaluate on another."""

    train_labels_arr = np.asarray(train_labels)
    test_labels_arr = np.asarray(test_labels)

    # One-vs-rest linear regression probe
    classes = np.unique(train_labels_arr)
    one_hot = np.eye(len(classes))[np.searchsorted(classes, train_labels_arr)]
    weights = np.linalg.pinv(train_features) @ one_hot

    logits = test_features @ weights
    predictions = classes[np.argmax(logits, axis=1)]
    acc = accuracy(test_labels_arr, predictions)
    return {"accuracy": acc}


__all__ = ["recall_at_k", "linear_probe_cross_sensor"]
