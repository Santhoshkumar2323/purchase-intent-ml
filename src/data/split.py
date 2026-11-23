import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import settings
from src.errors import ConfigError
from src.logs import get_logger
from src.schema import SPLIT_COLUMN, TARGET

log = get_logger(__name__)

MIN_ROWS = 500         
FALLBACK_TEST_SIZE = 0.25


def _usable(train: pd.DataFrame, test: pd.DataFrame) -> bool:
    big_enough = len(train) >= MIN_ROWS and len(test) >= MIN_ROWS
    both_classes = train[TARGET].nunique() == 2 and test[TARGET].nunique() == 2
    return big_enough and both_classes


def split_by_month(df: pd.DataFrame, train_months: list[str] | None = None):
    months = train_months or settings.data.train_months
    unknown = set(months) - set(df[SPLIT_COLUMN].unique())
    if unknown:  
        raise ConfigError(f"data.train_months has unknown months: {sorted(unknown)}")
    mask = df[SPLIT_COLUMN].isin(months)
    train, test = df[mask], df[~mask]
    if _usable(train, test):
        return train, test, "month"
    log.warning("Month split unusable (too small or one class); using random split.")
    train, test = train_test_split(
        df, test_size=FALLBACK_TEST_SIZE, stratify=df[TARGET], random_state=settings.data.seed
    )
    return train, test, "random"


def xy(df: pd.DataFrame):
    return df.drop(columns=[TARGET]), df[TARGET]