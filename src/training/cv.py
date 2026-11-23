import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.config import settings
from src.logs import get_logger
from src.training.models import make_estimator

log = get_logger(__name__)

SCORING = {"roc_auc": "roc_auc", "pr_auc": "average_precision"}


def _usable_folds(y) -> int:
    wanted = settings.training.cv_folds
    minority = int(y.value_counts().min())
    if minority < wanted:
        log.warning("Only %d minority rows; reducing CV folds from %d.", minority, wanted)
    return min(wanted, minority)


def cross_validate_model(X, y) -> dict:
    folds = _usable_folds(y)
    if folds < 2:
        return {"skipped": "not enough rows for cross-validation"}
    try:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=settings.data.seed)
        scores = cross_validate(
            make_estimator(y), X, y, cv=splitter, scoring=SCORING,
            n_jobs=1, error_score="raise",
        )
    except Exception as err:
        log.warning("Cross-validation skipped: %s", err)
        return {"skipped": str(err)}
    out = {"folds": folds}
    for name in SCORING:
        values = scores[f"test_{name}"]
        out[f"{name}_mean"] = round(float(np.mean(values)), 4)
        out[f"{name}_std"] = round(float(np.std(values)), 4)
    return out