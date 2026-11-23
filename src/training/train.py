from src.config import settings
from src.data.clean import clean
from src.data.load import load_raw
from src.data.split import split_by_month, xy
from src.errors import PipelineError
from src.evaluation.metrics import compute_metrics
from src.paths import artifact_path
from src.storage import save_json, save_model
from src.training.cv import cross_validate_model
from src.training.models import fit_model
from src.training.runs import append_run, build_metadata


def run() -> dict:
    version = settings.model.version
    train, test, method = split_by_month(clean(load_raw()))
    X_tr, y_tr = xy(train)
    X_te, y_te = xy(test)

    cv = cross_validate_model(X_tr, y_tr)     
    model, info = fit_model(X_tr, y_tr)
    test_metrics = compute_metrics(y_te, model.predict_proba(X_te)[:, 1])  

    split = {"method": method, "train_months": settings.data.train_months,
             "train_rows": len(train), "test_rows": len(test),
             "train_purchase_rate": round(float(y_tr.mean()), 4),
             "test_purchase_rate": round(float(y_te.mean()), 4)}
    meta = build_metadata(split, info, cv, test_metrics)

    save_model(model, artifact_path(version, "model.joblib"))
    save_json(meta, artifact_path(version, "metadata.json"))
    append_run(meta)
    return meta


def print_summary(meta: dict) -> None:
    model, cv, test = meta["model"], meta["cv"], meta["test_metrics"]
    print(f"\nVersion: {meta['version']} | model: {model['type']}")
    if model["fallback"]:
        print(f"WARNING: fell back from {model['requested']} ({model['fallback']})")
    if "roc_auc_mean" in cv:
        print(f"CV ({cv['folds']} folds): roc_auc {cv['roc_auc_mean']} +/- {cv['roc_auc_std']}, "
              f"pr_auc {cv['pr_auc_mean']} +/- {cv['pr_auc_std']}")
    else:
        print(f"CV skipped: {cv.get('skipped')}")
    print(f"Test (later months): roc_auc {test['roc_auc']}, pr_auc {test['pr_auc']}, "
          f"f1 {test['f1']}, recall {test['recall']}, precision {test['precision']}")


def main() -> None:
    try:
        meta = run()
    except (PipelineError, OSError) as err:
        raise SystemExit(f"Error: {err}")
    print_summary(meta)
    print(f"\nSaved to artifacts/{meta['version']}/ (model.joblib, metadata.json)")


if __name__ == "__main__":
    main()