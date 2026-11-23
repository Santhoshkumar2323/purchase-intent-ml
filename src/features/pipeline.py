from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import settings
from src.errors import ConfigError
from src.schema import BINARY, CATEGORICAL, NUMERIC, SPLIT_COLUMN

ALWAYS_DROP = {SPLIT_COLUMN}


def dropped_columns(extra: list[str] | None = None) -> set[str]:
    requested = set(settings.features.drop if extra is None else extra)
    valid = set(NUMERIC + BINARY + CATEGORICAL) | ALWAYS_DROP
    unknown = requested - valid
    if unknown:
        raise ConfigError(f"features.drop has unknown columns: {sorted(unknown)}")
    return requested | ALWAYS_DROP


def feature_columns(extra_drop: list[str] | None = None) -> tuple[list[str], list[str]]:
    dropped = dropped_columns(extra_drop)
    numeric = [c for c in NUMERIC + BINARY if c not in dropped]
    categorical = [c for c in CATEGORICAL if c not in dropped]
    return numeric, categorical


def build_preprocessor(extra_drop: list[str] | None = None) -> ColumnTransformer:
    numeric, categorical = feature_columns(extra_drop)
    parts = []
    if numeric:
        scaler = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
        parts.append(("num", scaler, numeric))
    if categorical:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        parts.append(("cat", encoder, categorical))
    if not parts:
        raise ConfigError("features.drop removed every feature; nothing left to train on.")
    return ColumnTransformer(parts, remainder="drop")