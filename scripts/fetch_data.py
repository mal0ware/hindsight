"""Download the daily Fama-French factor file into data/.

Source: the Kenneth R. French data library. The file is free to download and
needs no key or account. It is not committed to this repository.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import urllib.request
from pathlib import Path

from hindsight.data import FRENCH_URL, RAW_NAME, parse_french_daily

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--force", action="store_true", help="download even if the file exists")
    args = parser.parse_args()
    data_dir: Path = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / RAW_NAME
    if target.exists() and not args.force:
        print(f"{target} already present; use --force to download again")
    else:
        request = urllib.request.Request(FRENCH_URL, headers={"User-Agent": "research-script"})
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read()
        frame = parse_french_daily(raw)
        target.write_bytes(raw)
        vintage = {
            "url": FRENCH_URL,
            "downloaded_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "first_date": str(frame.index[0].date()),
            "last_date": str(frame.index[-1].date()),
            "rows": len(frame),
        }
        (data_dir / "data_vintage.json").write_text(json.dumps(vintage, indent=2) + "\n")
        print(
            f"saved {target} ({len(frame)} days, {vintage['first_date']} to {vintage['last_date']})"
        )


if __name__ == "__main__":
    main()
