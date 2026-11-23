import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.calibration.plots import calibration_curve_plot
from src.calibration.wrapper import CalibratedModel, fit_calibrator
from src.config import settings
from src.data.clean import clean
from src.data.load import load_raw
from src.data.split import split_by_month, xy
from src.errors import ConfigError, DataError, PipelineError
from src.paths import artifact_path
from src.schema import SPLIT_COLUMN, TARGET
from src.storage import save_json, save_model
from src.training.models import fit_model

MIN_ROWS = 200


def _split_train(train):
    months = settings.model.calibration_months
    unknown = set(months) - set(train[SPLIT_COLUMN].unique())
    if unknown:
        raise ConfigError(f"model.calibration_months must be training months: {sorted(unknown)}")
    mask = train[SPLIT_COLUMN].isin(months)
    fit_df, calib_df = train[~mask], train[mask]
    if min(len(fit_df), len(calib_df)) < MIN_ROWS or calib_df[TARGET].nunique() < 2:
        raise DataError("Too few rows to hold out for calibration; change model.calibration_months.")
    return fit_df, calib_df


def _summary(y, p) -> dict:
    return {"roc_auc": round(float(roc_auc_score(y, p)), 4),
            "brier": round(float(brier_score_loss(y, p)), 4),
            "mean_predicted": round(float(np.mean(p)), 4),
            "actual_rate": round(float(np.mean(y)), 4)}


def run() -> dict:
    version, method = settings.model.version, settings.model.calibration
    train, test, _ = split_by_month(clean(load_raw()))
    fit_df, calib_df = _split_train(train)
    X_fit, y_fit = xy(fit_df)
    X_cal, y_cal = xy(calib_df)
    X_te, y_te = xy(test)

    base, info = fit_model(X_fit, y_fit)
    calibrator = fit_calibrator(base.predict_proba(X_cal)[:, 1], y_cal, method)
    model = CalibratedModel(base, calibrator, method)
    raw_te = base.predict_proba(X_te)[:, 1]
    cal_te = model.predict_proba(X_te)[:, 1]

    report = {"method": method, "base_model": info["type"],
              "fit_rows": len(fit_df), "calibration_rows": len(calib_df),
              "calibration_months": settings.model.calibration_months,
              "before": _summary(y_te, raw_te), "after": _summary(y_te, cal_te)}
    save_model(model, artifact_path(version, "model.joblib"))
    save_json(report, artifact_path(version, "calibration.json"))
    calibration_curve_plot(y_te, raw_te, cal_te, version)
    return report


def main() -> None:
    try:
        report = run()
    except (PipelineError, OSError) as err:
        raise SystemExit(f"Error: {err}")
    print(f"\nMethod: {report['method']} | fit rows {report['fit_rows']}, "
          f"calibration rows {report['calibration_rows']}")
    print(f"{'':<8}{'roc_auc':>9}{'brier':>8}{'mean_pred':>11}{'actual':>8}")
    for key in ("before", "after"):
        m = report[key]
        print(f"{key:<8}{m['roc_auc']:>9}{m['brier']:>8}{m['mean_predicted']:>11}{m['actual_rate']:>8}")
    print(f"\nSaved calibrated model to artifacts/{settings.model.version}/model.joblib")


if __name__ == "__main__":
    main()