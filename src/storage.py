import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt

from src.paths import figure_path


def _ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_model(obj: Any, path: Path) -> Path:
    joblib.dump(obj, _ensure_parent(path))
    return path


def load_model(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run the training scripts first (python -m src.training.train)."
        )
    return joblib.load(path)


def save_json(data: dict, path: Path) -> Path:
    with open(_ensure_parent(path), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    return path


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the earlier pipeline steps first.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_figure(fig: plt.Figure, version: str, name: str) -> Path:
    path = _ensure_parent(figure_path(version, name))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path