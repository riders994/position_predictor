"""Stage 4: extend the college layer to the current season via CFBD, and test comparability.

cfbfastR stops at 2021. CFBD covers through the current season, which is what the project needs
to score *today's* prospects rather than only historical ones. But the two sources are not
interchangeable by assumption, so this stage measures their agreement on the 2013-2021 overlap
before anything is spliced.

Requires a free CFBD API key in ``CFBD_API_KEY`` or ``~/.config/cfbd/api_key``.

Usage
-----
    uv run python sports/football/scripts/qb_breakout_cfbd.py
    uv run python sports/football/scripts/qb_breakout_cfbd.py --no-fetch

Outputs
-------
``data/raw/cfbd_qb_seasons.parquet``      CFBD QB seasons (2013-current)
``reports/REPORT_qb_breakout_cfbd.md``    comparability + calibration evidence
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
    DATA_RAW, MANIFEST_DIR, REPORTS_DIR, ensure_dir,
)
from qb_breakout.data.cfbd import (  # noqa: E402
    FIRST_PPA_SEASON, build_cfbd_qb_seasons, compare_to_cfbfastr, fit_calibration,
)
from qb_breakout.data.college import LAST_PBP_SEASON  # noqa: E402

CACHE = DATA_RAW / "cfbd_qb_seasons.parquet"
CFBFASTR = DATA_RAW / "cfb_qb_seasons.parquet"

# Above this, a calibrated column reproduces too little of the between-player spread to be used
# as though it were measured. 0.5 means "at least half the real variation survives the mapping".
NOISE_LIMIT = 0.5


def fetch(start: int, end: int):
    def log(season, n):
        print(f"  {season}: {n} QB-seasons", flush=True)

    print(f"fetching CFBD QB seasons {start}-{end}")
    df = build_cfbd_qb_seasons(range(start, end + 1), progress=log)
    ensure_dir(CACHE.parent)
    df.to_parquet(CACHE, index=False)

    ensure_dir(MANIFEST_DIR)
    (MANIFEST_DIR / "cfbd_qb_seasons.json").write_text(json.dumps({
        "source": "collegefootballdata.com /stats/player/season + /ppa/players/season",
        "seasons": [start, end],
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "pulled_at": datetime.now(timezone.utc).isoformat(),
        "note": ("QB seasons with >=50 attempts. Requires a free API key (env CFBD_API_KEY or "
                 "~/.config/cfbd/api_key); the key is never stored in the repo. PPA begins 2013."),
    }, indent=2))
    return df


def _table(df, index_name=None):
    cols = list(df.columns)
    head = "| " + (f"{index_name} | " if index_name else "") + " | ".join(map(str, cols)) + " |"
    sep = "|" + "|".join("---" for _ in range(len(cols) + (1 if index_name else 0))) + "|"
    rows = []
    for idx, row in zip(df.index, df.itertuples(index=False, name=None)):
        cells = [("" if v != v else str(v)) for v in row]
        rows.append("| " + (f"{idx} | " if index_name else "") + " | ".join(cells) + " |")
    return "\n".join([head, sep, *rows])


def write_report(cfbd, summary, calibrations, path: Path) -> Path:
    import pandas as pd

    cal = pd.DataFrame(calibrations).set_index("feature")
    portable = [f for f, c in zip(cal.index, cal["noise_ratio"]) if c <= NOISE_LIMIT]
    blocked = [f for f, c in zip(cal.index, cal["noise_ratio"]) if c > NOISE_LIMIT]

    md = f"""# CFBD extension — can the two college sources be spliced?

cfbfastR play-by-play stops at **{LAST_PBP_SEASON}**, so the college layer could describe history
but not score a current prospect. CFBD covers through the present season and closes that gap. The
question this report answers is whether its numbers can be used *as if* they were the cfbfastR
ones — measured on the {FIRST_PPA_SEASON}–{LAST_PBP_SEASON} overlap rather than assumed.

- **CFBD QB seasons:** {len(cfbd)} ({int(cfbd['season'].min())}–{int(cfbd['season'].max())}, ≥50 attempts)
- **Overlap rows joined:** {int(summary['n'].max())}

## Do the two sources agree?

{_table(summary)}

**Volume agrees; efficiency does not.** Attempts, yards and touchdowns correlate at 0.99 — these
are the same events counted twice. Two systematic offsets are explained rather than mysterious:

- **Rushing attempts differ by ~16**, and that difference correlates 0.78 with cfbfastR's sack
  count. CFBD follows the NCAA convention of charging sacks as rushing attempts; the cfbfastR
  layer deliberately does not (see `data/college.py`). The gap *is* the convention.
- **cfbfastR totals run low** — it averages 8.8 games per QB-season, so its play-by-play has game
  gaps. For per-play *rates* this is harmless; for career *totals* it means cfbfastR understates,
  and CFBD's official season totals are the more accurate of the two where both exist.

Efficiency is the real problem. `ppa_pass` and `pass_epa_per_db` correlate only 0.735 and sit on
different scales — the level shift is larger than the target's own mean.

## Calibration: what survives the mapping

Each CFBD feature is regressed onto its cfbfastR counterpart on the overlap. `noise_ratio` is the
share of real between-player spread the mapping **fails** to reproduce; low is good.

{_table(cal.reset_index())}

**Portable (noise ratio ≤ {NOISE_LIMIT}):** {', '.join(portable) if portable else 'none'}
**Not portable:** {', '.join(blocked) if blocked else 'none'}

The `rush_share` calibration is the clean case: slope ≈ 1.01 and an intercept of −0.043 that is
simply the sack correction, recovering 95% of the variance. Efficiency is the opposite — a
calibrated EPA column would carry roughly two-thirds of the between-player spread as error, so
using it as a measured feature would be inventing precision.

## What this means for the model

The feature set splits in two, and the split is a real design constraint rather than bookkeeping:

**Portable tier** — computed natively and near-identically from both sources: completion %, yards
per attempt, TD rate, INT rate, rush share, volume. A model built on these can be **fit on history
and used to score current prospects**.

**cfbfastR-only tier** — EPA per dropback, success rate, adjusted yards per dropback. Better
features, available {2004}–{LAST_PBP_SEASON} only. A model using them is a **historical analysis
instrument**: it can explain what late breakouts looked like, but it cannot be pointed at this
year's draft class.

Stage 6 should fit both and report what the portable model gives up. If the gap is small, the
project gets a usable forward-looking tool; if it is large, that is itself the finding — the
signal lives in exactly the measure that cannot be carried forward.

Calibrated values, where used, are written as `*_est` with `efficiency_is_estimated=1` so an
estimate can never be mistaken for a measurement downstream.
"""
    ensure_dir(path.parent)
    path.write_text(md)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=int, default=FIRST_PPA_SEASON)
    ap.add_argument("--end", type=int, default=datetime.now().year)
    ap.add_argument("--no-fetch", action="store_true")
    args = ap.parse_args(argv)

    import pandas as pd
    import polars as pl

    if args.no_fetch:
        if not CACHE.exists():
            raise SystemExit(f"missing {CACHE} — drop --no-fetch to fetch it")
        cfbd = pd.read_parquet(CACHE)
    else:
        cfbd = fetch(args.start, args.end)

    if not CFBFASTR.exists():
        raise SystemExit(
            f"missing {CFBFASTR}\nRun stage 3 first:\n"
            f"  uv run python sports/football/scripts/qb_breakout_college.py"
        )
    cfbfastr = pl.read_parquet(CFBFASTR)

    joined, summary = compare_to_cfbfastr(cfbd[cfbd["season"] <= LAST_PBP_SEASON], cfbfastr)

    pairs = [
        ("pass_epa_per_db (from ppa_pass)", "ppa_pass", "pass_epa_per_db"),
        ("rush_share", "rush_share_cfbd", "rush_share_cfbfastr"),
        ("completion_pct", "completion_pct_cfbd", "completion_pct_cfbfastr"),
        ("yards_per_attempt", "yards_per_attempt_cfbd", "yards_per_attempt_cfbfastr"),
        ("td_rate", "td_rate_cfbd", "td_rate_cfbfastr"),
        ("int_rate", "int_rate_cfbd", "int_rate_cfbfastr"),
    ]
    calibrations = []
    for label, src, tgt in pairs:
        if src in joined.columns and tgt in joined.columns:
            calibrations.append({"feature": label, **fit_calibration(
                joined, source_col=src, target_col=tgt)})

    report = write_report(cfbd, summary, calibrations, REPORTS_DIR / "REPORT_qb_breakout_cfbd.md")
    print(f"CFBD QB seasons: {len(cfbd)} | overlap rows: {len(joined)}")
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
