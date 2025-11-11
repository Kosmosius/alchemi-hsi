import numpy as np


def retrieval_at_k(q: np.ndarray, keys: np.ndarray, gt: np.ndarray, k: int = 1) -> float:
    qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    kn = keys / (np.linalg.norm(keys, axis=1, keepdims=True) + 1e-12)
    sims = qn @ kn.T
    idx = np.argsort(-sims, axis=1)[:, :k]
    hits = (idx == gt[:, None]).any(axis=1).mean()
    return float(hits)
