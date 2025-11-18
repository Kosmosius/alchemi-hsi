from __future__ import annotations

from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score


class PRAUCIou(TypedDict):
    ap: float
    iou: float


def pr_auc_iou(
    y_true_mask: NDArray[np.bool_], score_map: NDArray[np.floating[Any]], thresh: float
) -> PRAUCIou:
    y: NDArray[np.int_] = y_true_mask.ravel().astype(int)
    s: NDArray[np.float64] = score_map.ravel().astype(np.float64)
    ap = float(average_precision_score(y, s))
    yhat = (score_map >= thresh).astype(int)
    inter = (yhat & y_true_mask).sum()
    union = (yhat | y_true_mask).sum()
    iou = float(inter / max(1, union))
    return {"ap": ap, "iou": iou}
