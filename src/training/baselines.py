from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import settings
from src.data.clean import clean
from src.data.load import load_raw
from src.data.split import split_by_month, xy
from src.errors import PipelineError
from src.evaluation.metrics import compute_metrics
from src.features.pipeline import build_preprocessor
from src.logs import get_logger
from src.paths import report_path
from src.storage import save_json

log = get_logger(__name__)


def build_baselines() -> dict[str, Pipeline]:
    weight = settings.training.class_weight
    weight = None if str(weight).lower() == "none" else weight
    return {
        "always_no_purchase": Pipeline(
            [("prep", build_preprocessor()), ("clf", DummyClassifier(strategy="prior"))]
        ),
        "logistic_regression": Pipeline([
            ("prep", build_preprocessor()),
            ("clf", LogisticRegression(max_iter=1000, class_weight=weight,
                                       random_state=settings.data.seed)),
        ]),
    }


def evaluate_baselines() -> dict:
    train, test, method = split_by_month(clean(load_raw()))
    X_tr, y_tr = xy(train)
    X_te, y_te = xy(test)
    results = {}
    for name, model in build_baselines().items():
        try:  # one failing baseline should not hide the other
            model.fit(X_tr, y_tr)
            results[name] = compute_metrics(y_te, model.predict_proba(X_te)[:, 1])
        except Exception as err:
            log.error("%s failed: %s", name, err)
            results[name] = {"error": str(err)}
    split_info = {"method": method, "train_rows": len(train), "test_rows": len(test),
                  "train_purchase_rate": round(float(y_tr.mean()), 4),
                  "test_purchase_rate": round(float(y_te.mean()), 4)}
    return {"split": split_info, "models": results}


def print_report(report: dict) -> None:
    print("\nSplit:", report["split"])
    print(f"\n{'model':<22}{'acc':>8}{'roc_auc':>9}{'pr_auc':>8}{'f1':>8}{'recall':>8}")
    for name, m in report["models"].items():
        if "error" in m:
            print(f"{name:<22}FAILED: {m['error']}")
            continue
        print(f"{name:<22}{m['accuracy']:>8}{m['roc_auc']:>9}{m['pr_auc']:>8}{m['f1']:>8}{m['recall']:>8}")


def main() -> None:
    try:
        report = evaluate_baselines()
    except PipelineError as err:
        raise SystemExit(f"Error: {err}")
    print_report(report)
    path = save_json(report, report_path(settings.model.version, "baseline_metrics.json"))
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()