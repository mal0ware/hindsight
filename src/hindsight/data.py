"""Loading the daily market return from the Kenneth French data library."""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import pandas as pd

FRENCH_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)
RAW_NAME = "F-F_Research_Data_Factors_daily_CSV.zip"


def parse_french_daily(raw_zip: bytes) -> pd.DataFrame:
    """Parse the zipped CSV into decimal daily returns indexed by date.

    Columns: ``mkt_rf`` (market return minus the bill rate) and ``rf`` (the
    bill rate). The file quotes percent; this divides by 100.
    """
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as archive:
        name = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
        text = archive.read(name).decode("latin-1")
    rows: list[tuple[str, float, float]] = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 5 and len(parts[0]) == 8 and parts[0].isdigit():
            rows.append((parts[0], float(parts[1]) / 100.0, float(parts[4]) / 100.0))
    if not rows:
        raise ValueError("no daily rows found in the French file")
    frame = pd.DataFrame(rows, columns=["date", "mkt_rf", "rf"])
    frame["date"] = pd.to_datetime(frame["date"], format="%Y%m%d")
    return frame.set_index("date").sort_index()


def load_french_daily(data_dir: Path) -> pd.DataFrame:
    """Read the file saved by ``scripts/fetch_data.py``."""
    path = data_dir / RAW_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run `uv run python scripts/fetch_data.py` first"
        )
    return parse_french_daily(path.read_bytes())


def sha256(data_dir: Path) -> str:
    return hashlib.sha256((data_dir / RAW_NAME).read_bytes()).hexdigest()
