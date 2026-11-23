import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

from src.storage import save_figure


def calibration_curve_plot(y_true, raw, calibrated, version, bins: int = 10):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect")
    for label, proba in (("Before calibration", raw), ("After calibration", calibrated)):
        actual, predicted = calibration_curve(y_true, proba, n_bins=bins, strategy="quantile")
        ax.plot(predicted, actual, marker="o", label=label)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Actual purchase rate")
    ax.set_title("Calibration on the later months")
    ax.legend()
    return save_figure(fig, version, "calibration_curve")