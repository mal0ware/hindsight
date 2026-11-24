"""Loading daily returns from the Kenneth French data library."""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import pandas as pd

BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
RAW_NAME = "F-F_Research_Data_Factors_daily_CSV.zip"
INDUSTRY_NAME = "10_Industry_Portfolios_daily_CSV.zip"
SIZE_NAME = "Portfolios_Formed_on_ME_Daily_CSV.zip"
FRENCH_URL = BASE_URL + RAW_NAME
RAW_FILES = (RAW_NAME, INDUSTRY_NAME, SIZE_NAME)

MARKET = "Market"
INDUSTRIES = (
    "NoDur",
    "Durbl",
    "Manuf",
    "Enrgy",
    "HiTec",
    "Telcm",
    "Shops",
    "Hlth",
    "Utils",
    "Other",
)
SIZE_DECILES = (
    "Lo 10",
    "Dec 2",
    "Dec 3",
    "Dec 4",
    "Dec 5",
    "Dec 6",
    "Dec 7",
    "Dec 8",
    "Dec 9",
    "Hi 10",
)
VALUE_WEIGHTED = "Average Value Weighted Returns -- Daily"
MISSING = (-99.99, -999.0)


def _csv_text(raw_zip: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as archive:
        name = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
        return archive.read(name).decode("latin-1")


def parse_french_daily(raw_zip: bytes) -> pd.DataFrame:
    """Parse the zipped factor CSV into decimal daily returns indexed by date.

    Columns: ``mkt_rf`` (market return minus the bill rate) and ``rf`` (the
    bill rate). The file quotes percent; this divides by 100.
    """
    rows: list[tuple[str, float, float]] = []
    for line in _csv_text(raw_zip).splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 5 and len(parts[0]) == 8 and parts[0].isdigit():
            rows.append((parts[0], float(parts[1]) / 100.0, float(parts[4]) / 100.0))
    if not rows:
        raise ValueError("no daily rows found in the French file")
    frame = pd.DataFrame(rows, columns=["date", "mkt_rf", "rf"])
    frame["date"] = pd.to_datetime(frame["date"], format="%Y%m%d")
    return frame.set_index("date").sort_index()


def parse_french_portfolios(raw_zip: bytes, section: str = VALUE_WEIGHTED) -> pd.DataFrame:
    """Parse one section of a zipped daily portfolio CSV into decimal returns.

    The portfolio files hold several tables one after another, each under a
    title line. This reads the table under ``section`` and stops at the first
    line after it that is not a daily row. The library marks missing values
    with -99.99 or -999; they become NaN.
    """
    lines = _csv_text(raw_zip).splitlines()
    try:
        begin = next(i for i, line in enumerate(lines) if line.strip() == section)
    except StopIteration:
        raise ValueError(f"section {section!r} not found in the French file") from None
    header = [part.strip() for part in lines[begin + 1].split(",")][1:]
    dates: list[str] = []
    rows: list[list[float]] = []
    for line in lines[begin + 2 :]:
        parts = [part.strip() for part in line.split(",")]
        if not (len(parts) == len(header) + 1 and len(parts[0]) == 8 and parts[0].isdigit()):
            break
        dates.append(parts[0])
        rows.append([float(part) for part in parts[1:]])
    if not rows:
        raise ValueError("no daily rows found in the French file")
    frame = pd.DataFrame(rows, columns=header, index=pd.to_datetime(dates, format="%Y%m%d"))
    frame = frame.where(~frame.isin(MISSING)) / 100.0
    frame.index.name = "date"
    return frame.sort_index()


def _read(data_dir: Path, name: str) -> bytes:
    path = data_dir / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run `uv run python scripts/fetch_data.py` first"
        )
    return path.read_bytes()


def load_french_daily(data_dir: Path) -> pd.DataFrame:
    """Read the factor file saved by ``scripts/fetch_data.py``."""
    return parse_french_daily(_read(data_dir, RAW_NAME))


def load_excess_returns(data_dir: Path) -> pd.DataFrame:
    """Daily excess returns of the market, ten industries and ten size deciles.

    The portfolio files hold total returns, so the bill rate from the factor
    file is subtracted. Only days present in all three files are kept, and a
    missing value in any of the 21 series is an error.
    """
    factors = load_french_daily(data_dir)
    industries = parse_french_portfolios(_read(data_dir, INDUSTRY_NAME))[list(INDUSTRIES)]
    sizes = parse_french_portfolios(_read(data_dir, SIZE_NAME))[list(SIZE_DECILES)]
    sizes = sizes.rename(columns={c: f"Size {c}" for c in sizes.columns})
    joined = pd.concat([factors, industries, sizes], axis=1, join="inner")
    out = joined.drop(columns=["mkt_rf", "rf"]).sub(joined["rf"], axis=0)
    out.insert(0, MARKET, joined["mkt_rf"])
    if out.isna().any().any():
        raise ValueError("missing values in the portfolio returns")
    return out


def sha256(data_dir: Path, name: str = RAW_NAME) -> str:
    return hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
