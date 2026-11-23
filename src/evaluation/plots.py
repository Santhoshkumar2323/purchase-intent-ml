import matplotlib

matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

from src.storage import save_figure

METRIC_KEYS = ["roc_auc", "pr_auc", "f1"]

def roc_pr_curves(y_true, proba, version):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    RocCurveDisplay.from_predictions(y_true, proba, ax=ax1)
    ax1.plot([0, 1], [0, 1], "--", color="grey")
    ax1.set_title("ROC curve")
    PrecisionRecallDisplay.from_predictions(y_true, proba, ax=ax2)
    ax2.set_title("Precision-recall curve")
    return save_figure(fig, version, "roc_pr_curves")


def confusion_matrix_plot(y_true, proba, version, threshold=0.5):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, (proba >= threshold).astype(int),
        display_labels=["No purchase", "Purchase"], ax=ax, colorbar=False,
    )
    ax.set_title(f"Confusion matrix (threshold {threshold})")
    return save_figure(fig, version, "confusion_matrix")


def comparison_bars(results: dict, version, name, title):
    labels, n = list(results), len(METRIC_KEYS)
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, key in enumerate(METRIC_KEYS):
        values = [results[label].get(key) or 0 for label in labels]
        ax.bar([j + i * width for j in range(len(labels))], values, width, label=key)
    ax.set_xticks([j + width * (n - 1) / 2 for j in range(len(labels))])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend()
    return save_figure(fig, version, name)