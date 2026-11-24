"""Download three daily files from the Kenneth R. French data library into data/.

The factor file gives the market excess return and the bill rate. The two
portfolio files give ten industry portfolios and the portfolios formed on
size. All are free to download and need no key or account. None is committed
to this repository.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import urllib.request
from pathlib import Path

from hindsight.data import (
    BASE_URL,
    RAW_FILES,
    RAW_NAME,
    parse_french_daily,
    parse_french_portfolios,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--force", action="store_true", help="download even if a file exists")
    args = parser.parse_args()
    data_dir: Path = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    vintage_path = data_dir / "data_vintage.json"
    vintage: dict[str, dict[str, object]] = {}
    if vintage_path.exists():
        stored = json.loads(vintage_path.read_text())
        # the first version of this script wrote one flat record for the factor file
        vintage = {RAW_NAME: stored} if "url" in stored else stored
    for name in RAW_FILES:
        target = data_dir / name
        if target.exists() and not args.force:
            print(f"{target} already present; use --force to download again")
            continue
        url = BASE_URL + name
        request = urllib.request.Request(url, headers={"User-Agent": "research-script"})
        with urllib.request.urlopen(request, timeout=120) as response:
            raw = response.read()
        frame = parse_french_daily(raw) if name == RAW_NAME else parse_french_portfolios(raw)
        target.write_bytes(raw)
        vintage[name] = {
            "url": url,
            "downloaded_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "first_date": str(frame.index[0].date()),
            "last_date": str(frame.index[-1].date()),
            "rows": len(frame),
        }
        print(f"saved {target} ({len(frame)} days to {vintage[name]['last_date']})")
    vintage_path.write_text(json.dumps(vintage, indent=2) + "\n")


if __name__ == "__main__":
    main()
