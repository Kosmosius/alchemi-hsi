import numpy as np

from alchemi.eval.retrieval import (
    compute_retrieval_at_k,
    random_retrieval_at_k,
    retrieval_summary,
    spectral_angle_deltas,
)


def test_retrieval_summary_beats_random():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(8, 6))
    keys = q + 0.05 * rng.normal(size=q.shape)
    gt = np.arange(q.shape[0])

    summary = retrieval_summary(q, keys, gt, ks=(1, 3))

    assert 0.7 <= summary["retrieval@1"] <= 1.0
    assert summary["retrieval@3"] >= summary["retrieval@1"]


def test_spectral_angle_deltas_distinguish_pairs():
    rng = np.random.default_rng(1)
    z_lab = rng.normal(size=(6, 5))
    z_sensor = z_lab + 0.1 * rng.normal(size=z_lab.shape)

    metrics = spectral_angle_deltas(z_lab, z_sensor)

    assert metrics["matched"] < metrics["mismatched"]
    assert metrics["delta"] > 0


def test_compute_retrieval_metrics_matches_random_baseline():
    rng = np.random.default_rng(2)
    z_lab = rng.normal(size=(10, 4))
    z_sensor = z_lab + 0.5 * rng.normal(size=z_lab.shape)
    metrics = compute_retrieval_at_k(z_lab, z_sensor, k=1)
    baseline = random_retrieval_at_k(len(z_lab), len(z_sensor), k=1)
    assert metrics.recall > baseline
