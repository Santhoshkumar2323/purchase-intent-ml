import pandas as pd

from src.config import settings
from src.errors import DataError
from src.logs import get_logger
from src.paths import raw_data_path
from src.schema import ALL_COLUMNS

log = get_logger(__name__)


def load_raw(filename: str | None = None) -> pd.DataFrame:
    path = raw_data_path(filename or settings.data.raw_filename)
    if not path.exists():
        raise DataError(f"Dataset not found at {path}. Run: python -m scripts.download_data")
    try:
        df = pd.read_csv(path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as err:
        raise DataError(f"Could not read {path} ({err}). Delete it and download again.") from err
    missing = [c for c in ALL_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(f"Dataset is missing expected columns: {missing}")
    log.info("Loaded %d rows, %d columns", *df.shape)
    return df