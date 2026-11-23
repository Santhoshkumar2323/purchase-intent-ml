import pandas as pd

from src.config import settings
from src.data.clean import clean
from src.data.load import load_raw
from src.data.split import split_by_month
from src.drift.categorical import max_shift, shares
from src.drift.plots import drift_chart
from src.drift.psi import proportions, psi
from src.drift.reference import build_reference, load_reference, save_reference
from src.errors import DataError, PipelineError
from src.paths import report_path
from src.storage import save_json


def _status(value: float, warn: float, alert: float) -> str:
    return "alert" if value >= alert else "warn" if value >= warn else "ok"


def compare(batch: pd.DataFrame, ref: dict) -> dict:
    cfg = settings.drift
    if len(batch) < cfg.min_batch_size:
        return {"error": f"Batch has {len(batch)} rows; need at least {cfg.min_batch_size}."}
    missing = [c for c in [*ref["numeric"], *ref["categorical"]] if c not in batch.columns]
    if missing:
        raise DataError(f"Batch is missing columns: {missing}")
    numeric = [
        {"feature": col, "psi": round(psi(spec["expected"], proportions(batch[col], spec["edges"])), 4)}
        for col, spec in ref["numeric"].items()
    ]
    for row in numeric:
        row["status"] = _status(row["psi"], cfg.psi_warn, cfg.psi_alert)
    categorical = []
    for col, expected in ref["categorical"].items():
        shift, category = max_shift(expected, shares(batch[col]))
        status = "alert" if shift >= cfg.cat_share_alert else "ok"
        categorical.append({"feature": col, "max_shift": shift, "category": category, "status": status})
    flagged = [r for r in numeric + categorical if r["status"] != "ok"]
    return {"batch_rows": len(batch), "numeric": numeric, "categorical": categorical,
            "flagged": len(flagged), "monitored": len(numeric) + len(categorical)}


def print_report(rep: dict) -> None:
    print(f"\nBatch rows: {rep['batch_rows']} | flagged {rep['flagged']} of {rep['monitored']} features")
    print("\nNumeric (PSI):")
    for r in sorted(rep["numeric"], key=lambda r: -r["psi"]):
        print(f"  {r['feature']:<26}{r['psi']:>8}  {r['status']}")
    print("\nCategorical (largest share shift):")
    for r in sorted(rep["categorical"], key=lambda r: -r["max_shift"]):
        print(f"  {r['feature']:<26}{r['max_shift']:>8}  {r['status']}  (category {r['category']})")


def main() -> None:
    version = settings.model.version
    try:
        train, test, _ = split_by_month(clean(load_raw()))
        save_reference(build_reference(train))
        rep = compare(test, load_reference())
        if "error" in rep:
            raise SystemExit(f"Error: {rep['error']}")
        save_json(rep, report_path(version, "drift_report.json"))
        drift_chart(rep, version)
    except (PipelineError, OSError) as err:
        raise SystemExit(f"Error: {err}")
    print_report(rep)
    print(f"\nSaved reference_stats.json to artifacts/{version}/, "
          f"drift_report.json and drift_report.png to reports/{version}/")


if __name__ == "__main__":
    main()