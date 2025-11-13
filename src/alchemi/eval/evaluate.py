import numpy as np

from .metrics_gas import pr_auc_iou
from .metrics_solids import banddepth_mae, macro_f1
from .retrieval import retrieval_at_k


def evaluate_solids(y_true, y_pred, nm=None, R_true=None, R_pred=None, windows=None):
    out = {"macro_f1": macro_f1(np.asarray(y_true), np.asarray(y_pred))}
    if nm is not None and R_true is not None and R_pred is not None and windows:
        out["banddepth_mae"] = banddepth_mae(nm, R_true, R_pred, windows)
    return out


def evaluate_gases(mask_true, score_map, thresh=0.5):
    return pr_auc_iou(np.asarray(mask_true), np.asarray(score_map), thresh)


def evaluate_alignment(emb_field, emb_lab, gt_index, k=1):
    return {
        f"retrieval@{k}": retrieval_at_k(
            np.asarray(emb_field), np.asarray(emb_lab), np.asarray(gt_index), k
        )
    }
