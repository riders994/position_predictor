"""Stage 2 of the QB-benching project: preseason features.

Everything an opening starter's club knew by **Sept 1** — prior-season production, tenure and
draft capital, the week-1 quarterback room, the club's prior year and whether the head coach is
new, plus what happened the last time he opened a season.

Usage
-----
    uv run python sports/football/scripts/qb_benching_features.py

Outputs
-------
``data/processed/qb_benching_features.parquet``     one row per opening starter, with features
``data/processed/qb_benching_blocks.json``          block -> columns map
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from position_predictor.utils.io import (  # noqa: E402
    DATA_PROCESSED, DATA_RAW, ensure_dir, write_parquet,
)
from qb_benching.features import build_features  # noqa: E402
from qb_benching.labels import game_starters, qb_depth  # noqa: E402
from qb_breakout.situation.regime import head_coach_by_team_season  # noqa: E402


def _read(name):
    import polars as pl

    path = DATA_RAW / f"{name}.parquet"
    if not path.exists():
        raise SystemExit(f"missing {path}\nRun stage 1 first, or `make -C sports/football fetch`.")
    return pl.read_parquet(path)


def main() -> None:
    import polars as pl

    from medstaff.data.teams import canonicalize

    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()

    cohort_path = DATA_PROCESSED / "qb_benching_cohort.parquet"
    if not cohort_path.exists():
        raise SystemExit(
            f"missing {cohort_path}\nRun stage 1 first:\n"
            "  uv run python sports/football/scripts/qb_benching_cohort.py")
    cohort = pl.read_parquet(cohort_path)

    schedules = _read("schedules")
    weekly = _read("weekly")
    rosters = _read("rosters")
    draft_picks = canonicalize(_read("draft_picks"), "team")
    depth = qb_depth(_read("depth_charts"), schedules)
    starters = game_starters(schedules)
    coach = canonicalize(head_coach_by_team_season(schedules), "team")

    frame, blocks = build_features(
        cohort, weekly=weekly, starters=starters, rosters=rosters, draft_picks=draft_picks,
        depth=depth, schedules=schedules, head_coach=coach)

    ensure_dir(DATA_PROCESSED)
    write_parquet(frame, DATA_PROCESSED / "qb_benching_features.parquet")
    (DATA_PROCESSED / "qb_benching_blocks.json").write_text(json.dumps(blocks, indent=2))

    n_cols = sum(len(v) for v in blocks.values())
    print(f"rows {frame.height} · features {n_cols} in {len(blocks)} blocks")
    for name, cols in blocks.items():
        miss = {c: round(frame[c].null_count() / frame.height, 3) for c in cols
                if frame[c].null_count()}
        print(f"  {name:11s} {len(cols):2d} cols" + (f"  nulls: {miss}" if miss else ""))
    print(f"wrote {DATA_PROCESSED / 'qb_benching_features.parquet'}")


if __name__ == "__main__":
    main()
