import pandas as pd
import streamlit as st

from src.config import settings
from src.paths import figure_path, report_path
from src.storage import load_json

STATUS_ICON = {"ok": "🟢 ok", "warn": "🟠 warn", "alert": "🔴 alert"}


def _table(rows: list[dict], sort_by: str) -> pd.DataFrame:
    df = pd.DataFrame(rows).sort_values(sort_by, ascending=False)
    df["status"] = df["status"].map(STATUS_ICON)
    return df


def render() -> None:
    version = settings.model.version
    try:
        report = load_json(report_path(version, "drift_report.json"))
    except (FileNotFoundError, ValueError):
        st.info("No drift report yet. Run: python -m src.drift.report")
        return
    st.subheader(f"{report['flagged']} of {report['monitored']} features flagged")
    st.caption(f"Compared across {report['batch_rows']} incoming sessions. "
               "Green means the data looks like training data, red means it has changed a lot.")
    chart = figure_path(version, "drift_report")
    if chart.exists():
        st.image(str(chart), use_container_width=True)
    left, right = st.columns(2)
    left.subheader("Numeric features (PSI)")
    left.dataframe(_table(report["numeric"], "psi"), hide_index=True,
                   use_container_width=True)
    right.subheader("Categorical features (share shift)")
    right.dataframe(_table(report["categorical"], "max_shift"), hide_index=True,
                    use_container_width=True)
    st.caption("Simulation: the later months (Sep to Dec) stand in for incoming traffic. "
               "Drift means the inputs changed; it does not by itself prove the model got worse.")