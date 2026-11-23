import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.decision.ev import expected_value
from src.storage import save_figure


def ev_chart(bands: list[dict], version: str):
    p = np.linspace(0, 1, 501)
    ev = expected_value(p)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(p, ev, color="tab:blue")
    ax.axhline(0, color="grey", linewidth=1)
    ax.fill_between(p, ev, 0, where=ev > 0, color="tab:green", alpha=0.25, label="Offer pays off")
    ax.fill_between(p, ev, 0, where=ev <= 0, color="tab:red", alpha=0.2, label="Offer loses money")
    for band in bands[:-1]:
        ax.axvline(band["max_prob"], linestyle="--", color="black", linewidth=1)
    ax.set_xlabel("Predicted purchase probability")
    ax.set_ylabel("Expected profit of showing an offer (Rs)")
    ax.set_title("Expected value of an offer vs purchase probability")
    ax.legend()
    return save_figure(fig, version, "ev_vs_probability")