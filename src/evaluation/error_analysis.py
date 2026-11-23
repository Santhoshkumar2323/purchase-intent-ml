import pandas as pd

from src.evaluation.metrics import compute_metrics
from src.schema import TARGET

GROUP_COLUMNS = ["Month", "VisitorType", "Weekend"]
MIN_ROWS = 30  


def by_group(test_df: pd.DataFrame, proba, column: str) -> list[dict]:
    df = test_df.assign(_proba=proba)
    rows = []
    for value, group in df.groupby(column):
        if len(group) < MIN_ROWS:
            continue
        m = compute_metrics(group[TARGET], group["_proba"])
        rows.append({
            column: str(value), "rows": len(group),
            "purchase_rate": round(float(group[TARGET].mean()), 4),
            "roc_auc": m["roc_auc"], "recall": m["recall"], "precision": m["precision"],
        })
    return rows


def analyse(test_df: pd.DataFrame, proba) -> dict:
    return {column: by_group(test_df, proba, column) for column in GROUP_COLUMNS}