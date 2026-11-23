import pandas as pd

from src.config import settings
from src.drift.categorical import shares
from src.drift.psi import make_edges, proportions
from src.features.pipeline import feature_columns
from src.paths import artifact_path
from src.storage import load_json, save_json

FILENAME = "reference_stats.json"


def monitored_columns() -> tuple[list[str], list[str]]:
    numeric, categorical = feature_columns()
    ignore = set(settings.drift.ignore)
    return ([c for c in numeric if c not in ignore],
            [c for c in categorical if c not in ignore])


def build_reference(train_df: pd.DataFrame) -> dict:
    numeric, categorical = monitored_columns()
    stats = {"rows": len(train_df), "numeric": {}, "categorical": {}}
    for col in numeric:
        edges = make_edges(train_df[col])
        stats["numeric"][col] = {
            "edges": edges,
            "expected": proportions(train_df[col], edges).round(6).tolist(),
        }
    for col in categorical:
        stats["categorical"][col] = shares(train_df[col])
    return stats


def save_reference(stats: dict) -> None:
    save_json(stats, artifact_path(settings.model.version, FILENAME))


def load_reference() -> dict:
    return load_json(artifact_path(settings.model.version, FILENAME))