import pandas as pd

from src.errors import DataError
from src.logs import get_logger
from src.schema import ALL_COLUMNS, CATEGORICAL, NUMERIC, TARGET

log = get_logger(__name__)

_BOOL_MAP = {"true": 1, "false": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0}

def _to_binary(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().map(_BOOL_MAP)


def _clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce").clip(lower=0)
    return df


def _clean_categorical(df: pd.DataFrame) -> pd.DataFrame:
    for col in CATEGORICAL:
        series = df[col]
        if pd.api.types.is_float_dtype(series):
            series = series.astype("Int64") 
        df[col] = series.astype("string").str.strip().fillna("Unknown").astype(str)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise DataError("Dataset is empty.")
    df = df[ALL_COLUMNS].copy()
    df[TARGET] = _to_binary(df[TARGET])
    df["Weekend"] = _to_binary(df["Weekend"])
    df["Month"] = df["Month"].astype(str).str.strip()
    before = len(df)
    df = df.dropna(subset=[TARGET]).drop_duplicates().reset_index(drop=True)
    log.info("Removed %d unusable or duplicate rows (%d left)", before - len(df), len(df))
    if df[TARGET].nunique() < 2:
        raise DataError("Target has only one class after cleaning; cannot train a classifier.")
    df[TARGET] = df[TARGET].astype(int)
    return _clean_categorical(_clean_numeric(df))