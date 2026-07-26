"""Stage 2: fetch the high-school recruiting layer and link it to the QB cohort.

Pulls ESPN recruiting classes (no API key required), keeps QB-ish prospects, and joins them to
the NFL careers built in stage 1. Writes a join-quality report — an unmatched QB is recorded as
unmatched, never silently dropped, because a biased match rate would quietly reshape the cohort.

Usage
-----
    uv run python sports/football/scripts/qb_breakout_recruiting.py
    uv run python sports/football/scripts/qb_breakout_recruiting.py --start 2006 --end 2025
    uv run python sports/football/scripts/qb_breakout_recruiting.py --no-fetch   # use cache

Outputs
-------
``data/raw/espn_recruits_qb.parquet``            raw QB-ish recruiting records
``data/processed/qb_breakout_recruiting.parquet`` cohort joined to recruiting profile
``reports/REPORT_qb_breakout_recruiting.md``     coverage and join-quality
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from position_predictor.utils.io import (  # noqa: E402
    DATA_PROCESSED, DATA_RAW, MANIFEST_DIR, REPORTS_DIR, ensure_dir,
)
from qb_breakout.data.recruiting import fetch_qb_recruits  # noqa: E402

# ESPN recruiting is effectively empty before 2006 (2002 returns 0 records, 2004 returns 2), so
# the layer starts there. That, not the NFL data, is what bounds the modelling window.
FIRST_CLASS = 2006
CACHE = DATA_RAW / "espn_recruits_qb.parquet"


def fetch(start: int, end: int):
    import pandas as pd

    def log(year, n):
        print(f"  {year}: {n} QB-ish recruits", flush=True)

    print(f"fetching ESPN recruiting classes {start}-{end}")
    df = fetch_qb_recruits(range(start, end + 1), progress=log)
    ensure_dir(CACHE.parent)
    df.to_parquet(CACHE, index=False)

    ensure_dir(MANIFEST_DIR)
    (MANIFEST_DIR / "espn_recruits_qb.json").write_text(json.dumps({
        "source": "ESPN core API /recruiting/{year}/athletes",
        "seasons": [start, end],
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "pulled_at": datetime.now(timezone.utc).isoformat(),
        "note": "QB-PP / QB-DT / ATH prospects only; measurables range-validated",
    }, indent=2))
    return pd.DataFrame(df)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=int, default=FIRST_CLASS)
    ap.add_argument("--end", type=int, default=datetime.now().year)
    ap.add_argument("--no-fetch", action="store_true", help="reuse the cached pull")
    args = ap.parse_args(argv)

    import pandas as pd

    from qb_breakout.data.link import link_recruits_to_cohort, recruiting_coverage_report

    careers_path = DATA_PROCESSED / "qb_breakout_careers.parquet"
    if not careers_path.exists():
        raise SystemExit(
            f"missing {careers_path}\nRun stage 1 first:\n"
            f"  uv run python sports/football/scripts/qb_breakout_cohort.py"
        )
    careers = pd.read_parquet(careers_path)

    if args.no_fetch:
        if not CACHE.exists():
            raise SystemExit(f"missing {CACHE} — drop --no-fetch to fetch it")
        recruits = pd.read_parquet(CACHE)
    else:
        recruits = fetch(args.start, args.end)

    linked = link_recruits_to_cohort(careers, recruits)
    ensure_dir(DATA_PROCESSED)
    linked.to_parquet(DATA_PROCESSED / "qb_breakout_recruiting.parquet", index=False)

    report = REPORTS_DIR / "REPORT_qb_breakout_recruiting.md"
    ensure_dir(report.parent)
    report.write_text(recruiting_coverage_report(linked, recruits))

    matched = int(linked["matched"].sum())
    print(f"recruits: {len(recruits)} | cohort: {len(careers)} | matched: {matched}")
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
