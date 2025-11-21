from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from .metrics_gas import PRAUCIou, pr_auc_iou
from .metrics_solids import banddepth_mae, macro_f1
from .retrieval import retrieval_at_k


def evaluate_solids(
    y_true: Iterable[Any],
    y_pred: Iterable[Any],
    nm: np.ndarray | None = None,
    R_true: np.ndarray | None = None,
    R_pred: np.ndarray | None = None,
    windows: list[tuple[float, float, float]] | None = None,
) -> dict[str, float]:
    out = {"macro_f1": macro_f1(np.asarray(y_true), np.asarray(y_pred))}
    if nm is not None and R_true is not None and R_pred is not None and windows:
        out["banddepth_mae"] = banddepth_mae(nm, R_true, R_pred, windows)
    return out


def evaluate_gases(mask_true: np.ndarray, score_map: np.ndarray, thresh: float = 0.5) -> PRAUCIou:
    return pr_auc_iou(np.asarray(mask_true), np.asarray(score_map), thresh)


def evaluate_alignment(
    emb_field: np.ndarray, emb_lab: np.ndarray, gt_index: np.ndarray, k: int = 1
) -> dict[str, float]:
    return {
        f"retrieval@{k}": retrieval_at_k(
            np.asarray(emb_field), np.asarray(emb_lab), np.asarray(gt_index), k
        )
    }
