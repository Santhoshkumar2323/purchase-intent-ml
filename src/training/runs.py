import csv
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as pkg_version

from src.config import settings
from src.features.pipeline import feature_columns
from src.logs import get_logger
from src.paths import runs_log_path

log = get_logger(__name__)

FIELDS = ["version", "trained_at", "model_type", "dropped",
          "cv_roc_auc", "test_roc_auc", "test_pr_auc", "test_f1"]


def _library_versions() -> dict:
    out = {}
    for name in ("scikit-learn", "xgboost", "pandas", "numpy"):
        try:
            out[name] = pkg_version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def build_metadata(split: dict, info: dict, cv: dict, test_metrics: dict) -> dict:
    numeric, categorical = feature_columns()
    return {
        "version": settings.model.version,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": settings.data.seed,
        "model": {**info, "requested": settings.model.type, "params": settings.model.params},
        "features": {"numeric": numeric, "categorical": categorical,
                     "dropped": sorted(settings.features.drop)},
        "split": split,
        "cv": cv,
        "test_metrics": test_metrics,
        "libraries": _library_versions(),
    }


def append_run(meta: dict) -> None:
    row = {
        "version": meta["version"], "trained_at": meta["trained_at"],
        "model_type": meta["model"]["type"],
        "dropped": "|".join(meta["features"]["dropped"]),
        "cv_roc_auc": meta["cv"].get("roc_auc_mean", ""),
        "test_roc_auc": meta["test_metrics"]["roc_auc"],
        "test_pr_auc": meta["test_metrics"]["pr_auc"],
        "test_f1": meta["test_metrics"]["f1"],
    }
    path = runs_log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            if is_new:
                writer.writeheader()
            writer.writerow(row)
    except OSError as err:
        log.warning("Could not update %s (%s). Close it if it is open elsewhere.", path, err)