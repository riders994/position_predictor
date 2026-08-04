"""Stage 2: turn weekly roster and injury-report rows into injury episodes.

An episode is one continuous injury spell. Every later stage aggregates these, so this stage
reports the distributions and censoring breakdown that make the construction auditable rather
than asking anyone to trust it.

Usage
-----
    uv run python sports/football/scripts/medstaff_episodes.py
    uv run python sports/football/scripts/medstaff_episodes.py --seasons 2021 2025

Outputs
-------
``data/processed/medstaff_panel.parquet``      player-week panel with the resolved week state
``data/processed/medstaff_episodes.parquet``   one row per spell, with recurrence attached
``reports/REPORT_medstaff_episodes.md``        counts, durations, recurrence, censoring
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
from medstaff.data import (  # noqa: E402
    EXCLUDED_SEASONS, FIRST_COMPARABLE_SEASON, FOCAL_GROUPS,
    load_injuries, load_rosters_weekly, load_schedules, roster_status_regime_table,
)
from medstaff.episodes import (  # noqa: E402
    PRIMARY_HORIZON, attach_recurrence, build_episodes, build_panel, recurrence_rates,
)

LATEST_SEASON = 2025


def _table(frame, *, floats: int = 3) -> str:
    df = frame.to_pandas() if hasattr(frame, "to_pandas") else frame

    def fmt(v):
        if v is None or (isinstance(v, float) and v != v):
            return "—"
        if isinstance(v, bool):
            return "yes" if v else "no"
        if isinstance(v, float):
            return f"{v:.{floats}f}"
        return str(v)

    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |"
            for row in df.itertuples(index=False, name=None)]
    return "\n".join([head, sep, *rows])


# A season-ending injury in week 1 costs a full season, so the longest spell in a five-season
# sample must approach the season length. The bye-week defect capped it at 13 and nothing errored
# for four stages — extremes are checked here, not just central tendency.
MIN_PLAUSIBLE_MAX_GAMES = 15
# Genuine mid-season releases of injured players are uncommon. The defect pushed this to 21%.
MAX_PLAUSIBLE_OFF_ROSTER_SHARE = 0.05


def sanity_checks(panel, episodes):
    """Loud checks against what the world must look like, not against the code's own logic.

    Every entry here exists because something silently wrong got through. Distributions looked
    healthy while the longest injury in five seasons was 13 games; tests passed because their
    fixtures assumed the very data shape that was wrong. So these are checked on *real* data.
    """
    import polars as pl

    issues = []
    max_games = int(episodes["games_missed"].max() or 0)
    if max_games < MIN_PLAUSIBLE_MAX_GAMES:
        issues.append(
            f"longest spell is only {max_games} games missed — a season-ending injury should "
            f"approach the season length; suspect absences are being truncated")

    off_share = (episodes.filter(pl.col("censor_reason") == "off_roster").height
                 / max(episodes.height, 1))
    if off_share > MAX_PLAUSIBLE_OFF_ROSTER_SHARE:
        issues.append(
            f"{off_share:.1%} of spells censor as off_roster — implausibly high; suspect "
            f"missing weeks are being read as the player leaving")

    # the panel must have no interior holes, or the builder cannot tell a bye from a release
    bounds = panel.group_by(["season", "gsis_id"]).agg(
        pl.col("week").min().alias("lo"), pl.col("week").max().alias("hi"),
        pl.col("week").n_unique().alias("have"))
    holes = bounds.filter(pl.col("hi") - pl.col("lo") + 1 > pl.col("have")).height
    if holes:
        issues.append(f"{holes:,} player-seasons still have interior missing weeks")

    return {"max_games_missed": max_games, "off_roster_share": off_share,
            "player_seasons_with_holes": holes, "issues": issues}


def build(seasons: tuple[int, ...], *, regime_seasons: tuple[int, ...] | None = None):
    injuries = load_injuries(DATA_RAW, seasons=seasons)
    rosters = load_rosters_weekly(DATA_RAW, seasons=seasons)
    schedules = load_schedules(DATA_RAW, seasons=seasons)

    panel = build_panel(rosters, injuries, schedules)
    episodes = build_episodes(panel)
    episodes = attach_recurrence(episodes, panel)

    # The comparability evidence spans seasons the episodes deliberately exclude, so it is
    # loaded separately — the point of the table is what it looks like OUTSIDE the window.
    regime = roster_status_regime_table(
        load_rosters_weekly(DATA_RAW, seasons=regime_seasons or tuple(range(2012, 2026)))
        .filter(__import__("polars").col("game_type") == "REG")
    )
    return {"panel": panel, "episodes": episodes, "regime": regime,
            "sanity": sanity_checks(panel, episodes)}


def write_report(result, path: Path, *, seasons: tuple[int, ...]) -> Path:
    import polars as pl

    eps, panel = result["episodes"], result["panel"]
    sanity = result["sanity"]
    n = eps.height
    resolved = eps.filter(pl.col("at_risk"))

    by_season = eps.group_by("season").agg(
        pl.len().alias("episodes"),
        pl.col("games_missed").mean().alias("mean_games_missed"),
        pl.col("censored").mean().alias("censored_share"),
    ).sort("season")

    by_group = eps.group_by("body_group").agg(
        pl.len().alias("episodes"),
        pl.col("games_missed").mean().alias("mean_games_missed"),
        pl.col("games_missed").median().alias("median_games_missed"),
        pl.col("used_reserve").mean().alias("ir_share"),
    ).sort("episodes", descending=True)

    by_pos = eps.group_by("position_group").agg(pl.len().alias("episodes")).sort(
        "episodes", descending=True)

    censor = eps.filter(pl.col("censored")).group_by("censor_reason").agg(
        pl.len().alias("episodes")).sort("episodes", descending=True)

    rec_focal = recurrence_rates(eps, by=["body_group"]).filter(
        pl.col("body_group").is_in(list(FOCAL_GROUPS))).sort("returns", descending=True)
    rec_all = recurrence_rates(eps)

    per_team = eps.group_by("team").agg(pl.len().alias("episodes")).sort("episodes")
    tm = per_team["episodes"]
    import math
    poisson_sd = math.sqrt(tm.mean())

    states = panel.group_by("week_state").agg(pl.len().alias("player_weeks")).sort(
        "player_weeks", descending=True)

    unknown = eps.filter(pl.col("body_group") == "unknown").height
    zero_miss = eps.filter(pl.col("games_missed") == 0).height

    md = f"""# Medical-staff grades — Stage 2: episodes

**{n:,} injury episodes** across {seasons[0]}–{seasons[-1]}
(excluding {", ".join(str(s) for s in EXCLUDED_SEASONS)}), built from
{panel.height:,} player-weeks.

## 1. How a spell is bounded

**Roster status is the availability signal, not snaps.** Players on `RES`, `INA`, `DEV` or
`CUT` take a snap in about 0.02% of player-weeks, so weekly status separates played from
did-not-play almost perfectly — while keying on `gsis_id` with no crosswalk loss and reaching
back to 2002. `snap_counts` keys on `pfr_player_id`, crosswalks at only 81.7% from 2012, and
would misread the ~21% of active player-weeks with zero snaps: a healthy scratch or a deep
backup is **available**, not injured.

Only an `ACT` week with no designation ends a spell. Inactive, reserve and bye weeks all
continue it, which absorbs report noise — a player listed in weeks 5 and 7 but merely inactive
in week 6 never "returned" in week 6.

{_table(states)}

## 2. ⚠️ Why the sample starts at 2021

Two `rosters_weekly` fields change meaning at 2021, and both would have corrupted every absence
measure here. `status_description_abbr` carries **no** reserve codes before 2021 — so reading IR
off it produced 2 episodes across 2012–2019 against ~1,000 per season after. And `status ==
"INA"` is barely populated earlier (2.8k rows across 2012–2019 vs 16.8k across 2021–2025) while
`ACT` falls 0.86 → 0.59: the gameday active/inactive split is simply absent from the earlier
data.

`status == "RES"` is the one stable signal, so **reserve is read from `status`, never from the
abbr codes**. This is the same shape as the sibling project's cfbfastR flag defect — an
unpopulated field aggregates to a clean zero rather than a null, so nothing errors and the
series quietly means something different on each side of the boundary.

{_table(result['regime'])}

**Reserve/COVID-19 (`R59`) is excluded.** It occurs in 2021 only — 725 player-weeks, exactly
zero in every other season — and carries `RES` status without being an injury. Left in, it
pushed 2021's reserve share to 0.46 against ~0.31 for 2022–2025, making every club look worse
at medicine in the first year of the five-year window.

## 3. Sanity checks against the world

Distributions can look healthy while the data is wrong. These check *extremes and impossibilities*
on real data, because the bye-week defect (§2) survived four stages behind a perfectly reasonable
mean of 3.3 games missed per spell.

- longest spell: **{sanity['max_games_missed']} games missed** (must approach season length)
- `off_roster` censoring: **{sanity['off_roster_share']:.1%}** (genuine in-season releases of
  injured players are uncommon)
- player-seasons with interior missing weeks: **{sanity['player_seasons_with_holes']:,}**

{chr(10).join('- ⚠️ ' + i for i in sanity['issues']) if sanity['issues'] else "**All clear.**"}

## 4. Episodes

{_table(by_season)}

### By body part

{_table(by_group)}

**{zero_miss:,} episodes ({zero_miss / n:.1%}) cost zero games** — knocks that were listed but
played through. They are kept, because incidence and recurrence both want them; the duration
model is the one that conditions on missed time.

**{unknown:,} episodes ({unknown / n:.1%}) have an `unknown` body part** — spells that opened on
a bare reserve week and never picked up a report row. Body part carries forward *within* a
spell but never across one, so an unrelated later stint cannot inherit an earlier injury.

### By position group

{_table(by_pos)}

## 5. Recurrence

Risk set is **returns, not episodes** — a spell that never resolved cannot recur, and counting
it would score an unresolved injury as a clean outcome. **{resolved.height:,} of {n:,} episodes
({resolved.height / n:.1%}) resolved** and are at risk.

The clock counts **games the player was available for**, not calendar weeks. A player returning
in week 17 has two games of exposure, not six weeks of it; counting calendar time would score a
late-season return as clean purely because the season ran out.

Overall, at the primary `{PRIMARY_HORIZON}` horizon:

{_table(rec_all)}

Focal body parts — the five with a plausible common-cause mechanism, and the subject of the
stage-5 signature analysis:

{_table(rec_focal)}

## 6. Censoring

{_table(censor)}

Censoring is not neutral here: a club can look good on return-to-play by having unrecovered
cases quietly censored, or by releasing injured players. Stage 4 therefore models
"returns at all" as its own outcome rather than folding it into duration.

## 7. Per-club dispersion

Episodes per club range **{tm.min():,} → {tm.max():,}** (mean {tm.mean():.0f},
sd {tm.std():.0f}). Poisson noise alone at this mean would give sd ≈ {poisson_sd:.0f}, so the
spread is **{tm.std() / poisson_sd:.1f}× wider than chance**.

That is the number this project exists to decompose — and most of it is not medicine. Roster
age, position mix, snap exposure, surface, and disclosure behaviour all sit inside it. Nothing
here is a grade; stage 4 builds the expectation that has to be subtracted first.
"""
    ensure_dir(path.parent)
    path.write_text(md)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seasons", nargs=2, type=int, metavar=("START", "END"),
                        default=[FIRST_COMPARABLE_SEASON, LATEST_SEASON],
                        help=f"inclusive season range "
                             f"(default {FIRST_COMPARABLE_SEASON}-{LATEST_SEASON}; absence "
                             f"measures are not comparable before {FIRST_COMPARABLE_SEASON})")
    args = parser.parse_args(argv)

    seasons = tuple(
        s for s in range(args.seasons[0], args.seasons[1] + 1) if s not in EXCLUDED_SEASONS
    )
    result = build(seasons)

    ensure_dir(DATA_PROCESSED)
    p1 = write_parquet(result["panel"], DATA_PROCESSED / "medstaff_panel.parquet")
    p2 = write_parquet(result["episodes"], DATA_PROCESSED / "medstaff_episodes.parquet")
    p3 = write_report(result, REPORTS_DIR / "REPORT_medstaff_episodes.md", seasons=seasons)

    eps = result["episodes"]
    sanity = result["sanity"]
    print(f"{eps.height:,} episodes from {result['panel'].height:,} player-weeks "
          f"({seasons[0]}–{seasons[-1]})")
    print(f"sanity: max {sanity['max_games_missed']} games missed · "
          f"off_roster {sanity['off_roster_share']:.1%} · "
          f"{sanity['player_seasons_with_holes']:,} panel holes")
    for issue in sanity["issues"]:
        print(f"  ⚠️  {issue}")
    for path in (p1, p2, p3):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
