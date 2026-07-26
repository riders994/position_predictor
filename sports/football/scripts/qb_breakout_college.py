"""Stage 3: build the college production layer and link it to the QB cohort.

Aggregates cfbfastR play-by-play into per-season and per-career QB production, then joins it to
the NFL careers from stage 1. This is the project's **primary** pre-NFL evidence — the
high-school layer was measured and set aside (``docs/QB_BREAKOUT_PLAN.md`` §2.3).

Usage
-----
    uv run python sports/football/scripts/qb_breakout_college.py
    uv run python sports/football/scripts/qb_breakout_college.py --start 2002 --end 2025
    uv run python sports/football/scripts/qb_breakout_college.py --no-fetch   # use cache

Outputs
-------
``data/raw/cfb_qb_seasons.parquet``            per QB college season
``data/processed/qb_breakout_college.parquet`` cohort joined to college career profile
``reports/REPORT_qb_breakout_college.md``      coverage and join quality
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from position_predictor.utils.io import (  # noqa: E402
    DATA_PROCESSED, DATA_RAW, MANIFEST_DIR, REPORTS_DIR, ensure_dir,
)
from qb_breakout.data.college import (  # noqa: E402
    FIRST_PBP_SEASON, add_career_features, build_college_qb_seasons,
)

CACHE = DATA_RAW / "cfb_qb_seasons.parquet"


def fetch(start: int, end: int, *, pbp_cache=None):
    """Aggregate play-by-play seasons into the QB-season table and cache it.

    Raw play-by-play is ~57 MB per season and is only needed to produce the aggregate, so by
    default it is staged in a temp directory and discarded rather than kept in ``data/``.
    """
    def log(season, n):
        print(f"  {season}: {n} QB-seasons", flush=True)

    print(f"aggregating cfbfastR play-by-play {start}-{end}")
    seasons = []
    for year in range(start, end + 1):
        try:
            frame = build_college_qb_seasons([year], cache_dir=pbp_cache, progress=log)
        except Exception as exc:  # a season may not be published yet
            print(f"  {year}: skipped ({type(exc).__name__}: {str(exc)[:80]})", flush=True)
            continue
        if frame.height:
            seasons.append(frame)

    import polars as pl

    df = pl.concat(seasons, how="diagonal_relaxed") if seasons else pl.DataFrame()
    ensure_dir(CACHE.parent)
    df.write_parquet(CACHE)

    ensure_dir(MANIFEST_DIR)
    (MANIFEST_DIR / "cfb_qb_seasons.json").write_text(json.dumps({
        "source": "cfbfastR-data pbp/parquet/play_by_play_{season}.parquet (public GitHub)",
        "seasons": [start, end],
        "rows": int(df.height),
        "cols": int(df.width),
        "pulled_at": datetime.now(timezone.utc).isoformat(),
        "note": "QB-seasons aggregated from play-by-play; >=50 dropbacks; sacks split from rushing",
    }, indent=2))
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=int, default=FIRST_PBP_SEASON)
    ap.add_argument("--end", type=int, default=datetime.now().year - 1)
    ap.add_argument("--no-fetch", action="store_true", help="reuse the cached aggregate")
    ap.add_argument("--keep-pbp", type=str, default=None,
                    help="directory to cache raw play-by-play in (default: temp, discarded)")
    args = ap.parse_args(argv)

    import polars as pl

    from qb_breakout.data.college_link import college_coverage_report, link_college_to_cohort

    careers_path = DATA_PROCESSED / "qb_breakout_careers.parquet"
    if not careers_path.exists():
        raise SystemExit(
            f"missing {careers_path}\nRun stage 1 first:\n"
            f"  uv run python sports/football/scripts/qb_breakout_cohort.py"
        )
    careers = pl.read_parquet(careers_path).to_pandas()

    if args.no_fetch:
        if not CACHE.exists():
            raise SystemExit(f"missing {CACHE} — drop --no-fetch to build it")
        qb_seasons = pl.read_parquet(CACHE)
    elif args.keep_pbp:
        qb_seasons = fetch(args.start, args.end, pbp_cache=args.keep_pbp)
    else:
        with tempfile.TemporaryDirectory(prefix="cfb_pbp_") as tmp:
            qb_seasons = fetch(args.start, args.end, pbp_cache=tmp)

    college_careers = add_career_features(qb_seasons).to_pandas()
    linked = link_college_to_cohort(careers, college_careers)

    ensure_dir(DATA_PROCESSED)
    linked.to_parquet(DATA_PROCESSED / "qb_breakout_college.parquet", index=False)

    report = REPORTS_DIR / "REPORT_qb_breakout_college.md"
    ensure_dir(report.parent)
    report.write_text(college_coverage_report(linked, qb_seasons, college_careers))

    matched = int(linked["college_matched"].sum())
    print(f"QB college seasons: {qb_seasons.height} | careers: {len(college_careers)}")
    print(f"cohort: {len(careers)} | matched: {matched}")
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
