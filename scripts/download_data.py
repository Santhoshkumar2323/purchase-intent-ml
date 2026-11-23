import io
import zipfile
from urllib.request import urlopen

import pandas as pd

from src.config import settings
from src.paths import raw_data_path

URL = (
    "https://archive.ics.uci.edu/static/public/468/"
    "online+shoppers+purchasing+intention+dataset.zip"
)
EXPECTED_ROWS, EXPECTED_COLS = 12330, 18


def download(filename: str) -> None:
    target = raw_data_path(filename)
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading from {URL} ...")
    with urlopen(URL, timeout=60) as resp:
        archive = zipfile.ZipFile(io.BytesIO(resp.read()))
    member = next(n for n in archive.namelist() if n.endswith(".csv"))
    target.write_bytes(archive.read(member))
    print(f"Saved to {target}")


def check(filename: str) -> None:
    df = pd.read_csv(raw_data_path(filename))
    ok = df.shape == (EXPECTED_ROWS, EXPECTED_COLS)
    print(f"Shape: {df.shape} -> {'OK' if ok else 'UNEXPECTED'}")
    print(f"Purchase rate: {df['Revenue'].mean():.1%}")
    print(f"Months: {sorted(df['Month'].unique())}")
    if not ok:
        raise SystemExit("Row/column count differs from the UCI description.")


def main() -> None:
    filename = settings.data.raw_filename
    if raw_data_path(filename).exists():
        print("File already exists, skipping download.")
    else:
        try:
            download(filename)
        except Exception as err: 
            raise SystemExit(
                f"Download failed ({err}).\nManual fallback: get the zip from the UCI "
                f"dataset page, unzip it, and put {filename} in data/raw/."
            )
    check(filename)


if __name__ == "__main__":
    main()