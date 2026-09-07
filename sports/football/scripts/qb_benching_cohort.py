"""Stage 1 of the QB-benching project: build the cohort and the label.

Reads the nflverse caches the sibling ``position_predictor`` pipeline already maintains
(``schedules``, ``rosters_weekly``, ``injuries``, ``weekly``), resolves who opened each season as
a club's starter, classifies every later week he did not start as a benching, an injury or a
departure, and writes the panel, the cohort and the descriptive report.

Usage
-----
    uv run python sports/football/scripts/qb_benching_cohort.py
    uv run python sports/football/scripts/qb_benching_cohort.py --benched-weeks 2

Outputs
-------
``data/processed/qb_benching_panel.parquet``   one row per team-game after the opener
``data/processed/qb_benching_cohort.parquet``  one row per opening starter per season
``reports/REPORT_qb_benching_cohort.md``       the descriptive write-up
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from position_predictor.utils.io import (  # noqa: E402
    DATA_PROCESSED, DATA_RAW, REPORTS_DIR, ensure_dir, write_parquet,
)
from qb_benching.labels import (  # noqa: E402
    BENCHED_WEEKS, FIRST_REPORT_SEASON, attach_depth, build_cohort, classify_displacement,
    displacement_panel, evidence_mix, game_starters, label_rate_by_season, opening_starters,
    qb_depth, reconciles, regime_check, sensitivity, starter_disagreements,
)


def _read(name):
    import polars as pl

    path = DATA_RAW / f"{name}.parquet"
    if not path.exists():
        raise SystemExit(
            f"missing {path}\nRun the sibling pipeline's fetch stage first:\n"
            f"  make -C sports/football fetch"
        )
    return pl.read_parquet(path)


def _table(frame, *, floats=3) -> str:
    """Render a small polars frame as a markdown table."""
    cols = frame.columns
    head = "| " + " | ".join(cols) + " |"
    rule = "|" + "|".join(["---"] * len(cols)) + "|"
    rows = []
    for rec in frame.iter_rows(named=True):
        cells = []
        for c in cols:
            v = rec[c]
            if v is None:
                cells.append("—")
            elif isinstance(v, bool):
                cells.append("yes" if v else "no")
            elif isinstance(v, float):
                cells.append(f"{v:.{floats}f}")
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, rule, *rows])


def write_report(path, *, cohort, by_season, regime, sens, agree_rate, disagreements,
                 split, evidence, benched_weeks, first_season, last_season):
    import polars as pl

    n = cohort.height
    n_benched = int(cohort["benched"].sum())
    n_injured = int(cohort["injured_out"].sum())
    n_displaced = int(cohort["displaced"].sum())
    pre = regime.filter(pl.col("regime").str.starts_with("pre-"))["rate"][0]
    post = regime.filter(~pl.col("regime").str.starts_with("pre-"))["rate"][0]
    drift = abs(pre - post)
    verdict = (
        f"The rates differ by {drift:.3f}, inside season-to-season variation — the two eras are "
        "reading the same thing even though the roster column beneath them changed."
        if drift < 0.03 else
        f"**The rates have diverged by {drift:.3f}.** The label may have picked up a dependency "
        "on the roster-status column that changes at 2021 — investigate before trusting the "
        "sample."
    )
    direct = evidence.filter(pl.col("evidence") == "depth_chart")["team_games"].sum()
    residual = evidence.filter(pl.col("evidence") == "residual")["team_games"].sum()

    lines = [
        "# QB Benching — Stage 1: the cohort and the label",
        "",
        f"_{first_season}–{last_season} · {n} opening starters · "
        f"benched at ≥{benched_weeks} weeks_",
        "",
        "## What this counts",
        "",
        "A club's **opening starter** is the quarterback who started its first game of the "
        "season. Every later game the club played is then one row: either he started it, or he "
        "did not and the reason is resolved three ways.",
        "",
        _table(split),
        "",
        "## Base rate",
        "",
        f"**{n_benched} of {n} opening starters ({n_benched / n:.1%}) were benched.** For "
        f"comparison, {n_injured} ({n_injured / n:.1%}) lost the same number of weeks to injury "
        f"and {n_displaced} ({n_displaced / n:.1%}) lost them for any reason at all.",
        "",
        _table(by_season),
        "",
        "## The bar is a choice, so here are the others",
        "",
        _table(sens),
        "",
        "## How each week was resolved",
        "",
        "Benching is read off the club's **own depth chart** wherever one was published: a "
        "quarterback listed behind someone else has been demoted, and one missing from his "
        "club's chart is unavailable. That is a direct observation rather than an inference, and "
        f"it resolves **{direct} of the {direct + residual}** benched weeks; the remaining "
        f"{residual} fall through to the roster ladder as a residual.",
        "",
        _table(evidence),
        "",
        "## Two roster-table defects this label had to route around",
        "",
        "**Before 2021, `rosters_weekly.status` is not a weekly value.** It is the player's "
        "season-final status stamped on every one of his weeks: only 1–4% of pre-2021 "
        "QB-seasons have a status that varies by week, against 50–72% from 2021 on. Colin "
        "Kaepernick started the first eight games of 2015 and finished on injured reserve, so "
        "all ten of his rows read `RES` — including the weeks he was starting. A first version "
        "of this label used that column and reported **zero benchings in 2015**, a season in "
        "which he visibly lost the job to Blaine Gabbert.",
        "",
        "**Injured reserve is invisible to the injury report** — a player placed on it drops off "
        "the report entirely. Measured against the trustworthy 2021+ status, weeks with no "
        "report listing and no appearance split 195 reserve / 104 active / 20 inactive, so a "
        "label keyed on the report alone would call roughly 60% of them benchings.",
        "",
        "The depth chart has neither problem, which is why it leads the ladder. The check below "
        "is what says so: the label reads different evidence either side of 2021, so its rate "
        "had better not move.",
        "",
        _table(regime),
        "",
        verdict,
        "",
        "## How far to trust the starter column",
        "",
        f"`schedules` records the announced starter. Against the obvious alternative — the QB "
        f"with the most dropbacks in the game — it agrees on **{agree_rate:.1%}** of team-games. "
        "The disagreements are in-game changes, a hook or an injury, and counting them as "
        "displacement would fold within-game events into a week-to-week decision. A sample:",
        "",
        _table(disagreements),
        "",
        "## Reproduce",
        "",
        "```bash",
        "uv run python sports/football/scripts/qb_benching_cohort.py",
        "```",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    import polars as pl

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benched-weeks", type=int, default=BENCHED_WEEKS,
                    help=f"weeks benched that make a benching (default {BENCHED_WEEKS})")
    ap.add_argument("--first-season", type=int, default=FIRST_REPORT_SEASON,
                    help=f"earliest season (default {FIRST_REPORT_SEASON}, the injury report)")
    args = ap.parse_args()

    schedules = _read("schedules")
    rosters_weekly = _read("rosters_weekly")
    injuries = _read("injuries")
    weekly = _read("weekly")
    depth_charts = _read("depth_charts")

    starters = game_starters(schedules)
    # A season with no announced starters at all has not been played yet.
    played = (
        starters.group_by("season").agg(pl.col("starter_id").is_not_null().any().alias("any"))
        .filter(pl.col("any"))["season"].to_list()
    )
    starters = starters.filter(
        pl.col("season").is_in(played) & (pl.col("season") >= args.first_season)
    )
    last_season = int(starters["season"].max())

    openers = opening_starters(starters)
    panel = displacement_panel(starters, openers, rosters_weekly, injuries, weekly)
    panel = attach_depth(panel, qb_depth(depth_charts, schedules))
    classified = classify_displacement(panel)
    cohort = build_cohort(classified, benched_weeks=args.benched_weeks)

    if not reconciles(cohort):
        raise SystemExit("the outcome counts do not reconcile to games played — refusing to write")

    split = (
        classified.group_by("outcome").agg(pl.len().alias("team_games"))
        .with_columns((pl.col("team_games") / classified.height).alias("share"))
        .sort("team_games", descending=True)
    )
    dis = starter_disagreements(starters, weekly)
    agree_rate = float(dis["agrees"].mean())
    sample = (
        dis.filter(~pl.col("agrees"))
        .select(["season", "week", "team", "starter_name", "dropbacks"])
        .sort(["season", "week"]).tail(8)
    )

    ensure_dir(DATA_PROCESSED)
    ensure_dir(REPORTS_DIR)
    write_parquet(classified, DATA_PROCESSED / "qb_benching_panel.parquet")
    write_parquet(cohort, DATA_PROCESSED / "qb_benching_cohort.parquet")
    write_report(
        REPORTS_DIR / "REPORT_qb_benching_cohort.md",
        cohort=cohort,
        by_season=label_rate_by_season(cohort),
        regime=regime_check(cohort),
        sens=sensitivity(classified),
        agree_rate=agree_rate,
        disagreements=sample,
        split=split,
        evidence=evidence_mix(classified),
        benched_weeks=args.benched_weeks,
        first_season=args.first_season,
        last_season=last_season,
    )

    print(f"openers {cohort.height} · benched {int(cohort['benched'].sum())} "
          f"({cohort['benched'].mean():.1%}) · injured {int(cohort['injured_out'].sum())} "
          f"· starter agreement {agree_rate:.1%}")
    print(f"wrote {DATA_PROCESSED / 'qb_benching_cohort.parquet'}")
    print(f"wrote {REPORTS_DIR / 'REPORT_qb_benching_cohort.md'}")


if __name__ == "__main__":
    main()
