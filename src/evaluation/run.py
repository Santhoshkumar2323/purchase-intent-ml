from src.config import settings
from src.data.clean import clean
from src.data.load import load_raw
from src.data.split import split_by_month, xy
from src.errors import PipelineError
from src.evaluation import plots
from src.evaluation.error_analysis import analyse
from src.evaluation.leakage import pagevalues_test
from src.evaluation.metrics import compute_metrics
from src.paths import artifact_path, report_path
from src.storage import load_json, load_model, save_json


def _baselines(version: str) -> dict:
    try:
        return load_json(report_path(version, "baseline_metrics.json"))["models"]
    except (OSError, KeyError, ValueError):
        return {}  


def evaluate() -> dict:
    version = settings.model.version
    train, test, _ = split_by_month(clean(load_raw()))
    X_tr, y_tr = xy(train)
    X_te, y_te = xy(test)

    model = load_model(artifact_path(version, "model.joblib"))
    proba = model.predict_proba(X_te)[:, 1]
    metrics = compute_metrics(y_te, proba)

    plots.roc_pr_curves(y_te, proba, version)
    plots.confusion_matrix_plot(y_te, proba, version)
    label = settings.model.type
    plots.comparison_bars({**_baselines(version), label: metrics}, version,
                          "model_comparison", "Models on the later months")

    leak = pagevalues_test(X_tr, y_tr, X_te, y_te, metrics)
    if "without" in leak:
        plots.comparison_bars({"With PageValues": leak["with"],
                               "Without PageValues": leak["without"]},
                              version, "pagevalues_leakage_test", "PageValues leakage test")

    report = {"model": metrics, "errors": analyse(test, proba), "pagevalues_test": leak}
    save_json(report, report_path(version, "metrics.json"))
    return report


def print_summary(report: dict) -> None:
    print("\nOverall:", report["model"])
    leak = report["pagevalues_test"]
    if "without" in leak:
        print(f"ROC-AUC with PageValues {leak['with']['roc_auc']} | "
              f"without {leak['without']['roc_auc']}")
    else:
        print("PageValues test:", leak.get("skipped"))
    print("\nBy month (roc_auc, recall):")
    for row in report["errors"]["Month"]:
        print(f"  {row['Month']:<5} rows {row['rows']:>5}  buy-rate {row['purchase_rate']:<7}"
              f" auc {row['roc_auc']}  recall {row['recall']}")


def main() -> None:
    try:
        report = evaluate()
    except (PipelineError, OSError) as err:
        raise SystemExit(f"Error: {err}")
    print_summary(report)
    print(f"\nSaved to reports/{settings.model.version}/ (metrics.json, figures/)")


if __name__ == "__main__":
    main()