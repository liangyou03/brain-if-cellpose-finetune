from __future__ import annotations

import numpy as np
from cellpose import metrics


def summarize_metrics(masks_true, masks_pred, threshold: float = 0.5):
    ap, tp, fp, fn = metrics.average_precision(masks_true, masks_pred, threshold=[threshold])
    if ap.ndim == 2:
        ap = ap[:, 0]
        tp = tp[:, 0]
        fp = fp[:, 0]
        fn = fn[:, 0]
    tp_sum = float(np.sum(tp))
    fp_sum = float(np.sum(fp))
    fn_sum = float(np.sum(fn))
    precision = tp_sum / (tp_sum + fp_sum + 1e-8)
    recall = tp_sum / (tp_sum + fn_sum + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    return {
        "ap50": float(np.mean(ap)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp_sum),
        "fp": int(fp_sum),
        "fn": int(fn_sum),
    }
