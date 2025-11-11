import numpy as np
from sklearn.metrics import average_precision_score


def pr_auc_iou(y_true_mask: np.ndarray, score_map: np.ndarray, thresh: float) -> dict:
    y = y_true_mask.ravel().astype(int)
    s = score_map.ravel().astype(float)
    ap = float(average_precision_score(y, s))
    yhat = (score_map >= thresh).astype(int)
    inter = (yhat & y_true_mask).sum()
    union = (yhat | y_true_mask).sum()
    iou = float(inter / max(1, union))
    return {"ap": ap, "iou": iou}
