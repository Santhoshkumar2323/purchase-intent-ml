import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import settings
from src.storage import save_figure

COLORS = {"ok": "tab:green", "warn": "tab:orange", "alert": "tab:red"}


def _bars(ax, rows, key, title, lines):
    rows = sorted(rows, key=lambda r: r[key])
    ax.barh([r["feature"] for r in rows], [r[key] for r in rows],
            color=[COLORS[r["status"]] for r in rows])
    for x, style in lines:
        ax.axvline(x, linestyle=style, color="black", linewidth=1)
    ax.set_title(title)


def drift_chart(report: dict, version: str):
    cfg = settings.drift
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    _bars(ax1, report["numeric"], "psi", "Numeric features: PSI",
          [(cfg.psi_warn, ":"), (cfg.psi_alert, "--")])
    _bars(ax2, report["categorical"], "max_shift", "Categorical: largest share shift",
          [(cfg.cat_share_alert, "--")])
    fig.suptitle("Drift: later months vs training months (green ok, orange warn, red alert)")
    fig.tight_layout()
    return save_figure(fig, version, "drift_report")