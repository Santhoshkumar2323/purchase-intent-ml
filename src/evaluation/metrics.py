import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)


def compute_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    pred = (proba >= threshold).astype(int)
    out = {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": None,
        "pr_auc": None,
    }
    if len(np.unique(y_true)) == 2:  
        out["roc_auc"] = roc_auc_score(y_true, proba)
        out["pr_auc"] = average_precision_score(y_true, proba)
    return {k: None if v is None else round(float(v), 4) for k, v in out.items()}