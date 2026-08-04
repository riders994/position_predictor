"""Stage 3: the exposure table — everything the medical staff does not control.

Stage 4 measures observed minus expected, so this stage assembles the expected side's inputs:
who was at risk, for how much football, under what conditions, at what age. Nothing here is an
outcome and nothing here is a grade.

Usage
-----
    uv run python sports/football/scripts/medstaff_exposure.py
    uv run python sports/football/scripts/medstaff_exposure.py --seasons 2021 2025

Outputs
-------
``data/processed/medstaff_exposure.parquet``   player-week risk set with covariates
``data/processed/medstaff_history.parquet``    per player-season-bodygroup prior-injury weeks
``reports/REPORT_medstaff_exposure.md``        covariate coverage and what it buys
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
    EXCLUDED_SEASONS, FIRST_COMPARABLE_SEASON, FIRST_INJURY_SEASON,
    load_injuries, load_raw, load_rosters_weekly, load_schedules,
)
from medstaff.exposure import (  # noqa: E402
    build_risk_set, club_home_surface, game_context, id_crosswalk, injury_history,
    player_attributes, snap_exposure,
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


def build(seasons: tuple[int, ...]):
    import polars as pl

    panel = pl.read_parquet(DATA_PROCESSED / "medstaff_panel.parquet")
    episodes = pl.read_parquet(DATA_PROCESSED / "medstaff_episodes.parquet")
    rosters = load_rosters_weekly(DATA_RAW, seasons=seasons)
    schedules = load_schedules(DATA_RAW, seasons=seasons)
    snaps_raw = load_raw(DATA_RAW, "snap_counts")
    ids = load_raw(DATA_RAW, "ids")
    players = load_raw(DATA_RAW, "players")

    # History reaches back to the first injury season, not the comparable window: the report is
    # comparable from 2009 even though the roster fields are not.
    all_injuries = load_injuries(
        DATA_RAW, seasons=tuple(range(FIRST_INJURY_SEASON, LATEST_SEASON + 1))
    )

    risk = build_risk_set(panel, episodes)
    xw = id_crosswalk(players, ids, rosters)
    snaps = snap_exposure(snaps_raw.filter(pl.col("season").is_in(list(seasons))), xw)
    attrs = player_attributes(rosters)
    ctx = game_context(schedules)
    home = club_home_surface(schedules)
    hist_overall, hist_group = injury_history(all_injuries, seasons=seasons)

    exposure = (
        risk.join(ctx.drop("team_played"), on=["season", "week", "team"], how="left")
        .join(home, on=["season", "team"], how="left")
        .join(attrs.drop("position", "position_group", "side"), on=["season", "gsis_id"],
              how="left")
        .join(snaps, on=["season", "week", "gsis_id"], how="left")
        .join(hist_overall, on=["season", "gsis_id"], how="left")
        .with_columns(
            pl.col("prior_designated_weeks").fill_null(0),
            pl.col("prior_injury_seasons").fill_null(0),
        )
    )
    return {
        "exposure": exposure, "history_group": hist_group,
        "snap_match": snaps, "risk": risk, "panel": panel,
    }


def write_report(result, path: Path, *, seasons: tuple[int, ...]) -> Path:
    import polars as pl

    exp = result["exposure"]
    panel = result["panel"]

    coverage = pl.DataFrame({
        "covariate": [
            "surface (game)", "home_surface (club)", "rest_days", "indoor",
            "age", "years_exp", "bmi", "snaps", "snap_share_3wk",
            "prior_designated_weeks", "weeks_since_return",
        ],
        "non_null_share": [
            1 - exp["surface"].null_count() / exp.height,
            1 - exp["home_surface"].null_count() / exp.height,
            1 - exp["rest_days"].null_count() / exp.height,
            1 - exp["indoor"].null_count() / exp.height,
            1 - exp["age"].null_count() / exp.height,
            1 - exp["years_exp"].null_count() / exp.height,
            1 - exp["bmi"].null_count() / exp.height,
            1 - exp["snaps"].null_count() / exp.height,
            1 - exp["snap_share_3wk"].null_count() / exp.height,
            1 - exp["prior_designated_weeks"].null_count() / exp.height,
            1 - exp["weeks_since_return"].null_count() / exp.height,
        ],
    })

    surf = exp.group_by("surface").agg(pl.len().alias("player_weeks")).sort(
        "player_weeks", descending=True)
    home_surf = exp.select(["season", "team", "home_surface"]).unique().group_by(
        "home_surface").agg(pl.col("team").n_unique().alias("clubs")).sort(
        "clubs", descending=True)
    rest = exp.group_by("short_week").agg(pl.len().alias("player_weeks")).sort("short_week")
    by_pos = exp.group_by("position_group").agg(
        pl.len().alias("risk_weeks"),
        pl.col("age").mean().alias("mean_age"),
        pl.col("snap_share").mean().alias("mean_snap_share"),
    ).sort("risk_weeks", descending=True)

    snap_by_pos = exp.group_by("position_group").agg(
        (1 - pl.col("snaps").is_null().mean()).alias("snap_coverage"),
        pl.col("snap_share").mean().alias("mean_snap_share"),
    ).sort("snap_coverage", descending=True)

    hist = exp.select("prior_designated_weeks")
    snap_cov = 1 - exp["snaps"].null_count() / exp.height

    md = f"""# Medical-staff grades — Stage 3: exposure and confounders

The risk set is **{exp.height:,} player-weeks** across {seasons[0]}–{seasons[-1]}, drawn from
{panel.height:,} panel rows. Nothing here is an outcome and nothing here is a grade — this is the
input side of the observed-minus-expected the grades are built from.

## 1. What the risk set excludes, and why

A player-week only counts as exposure if a new injury could have *started* in it:

- **Weeks already inside an open spell are dropped.** Counting them would turn one long absence
  into many weeks of injury-free exposure, which is exactly backwards.
- **Practice-squad weeks are dropped** — not exposure to an NFL game.
- **Byes are dropped** — the club did not play.

## 2. Covariate coverage

{_table(coverage)}

**Snap coverage is {snap_cov:.1%}.** `snap_counts` keys on `pfr_player_id`, so it needs a
crosswalk to `gsis_id`, and the obvious sources are biased in the one direction that would have
corrupted this project:

- `rosters_weekly.pfr_id` is null for **99.8% of offensive-line rows**, against 9–24% elsewhere
- `ids` is a *fantasy* table — of 352 distinct linemen in one season's snap counts it resolves
  **two**

Either would have left snap-workload covariates present for skill players and absent for
linemen. Position correlates with body part, and body part is exactly what the stage-5 signature
analysis compares, so position-biased missingness would have looked like a finding. The
crosswalk therefore comes from `players`, which is ~12% null for linemen and ~11% for skill
players — missing at roughly the same rate everywhere.

{_table(snap_by_pos)}

Snaps remain an *intensity* covariate and never the availability signal — the roster does that
job at full coverage — and every snap-derived column is null-safe.

## 3. Conditions the club does not choose

Surface is free text and dirty — `"grass "` with a trailing space is a distinct value from
`"grass"`, and `""` means missing — so it is normalised to grass/turf, with turf brands
collapsed because the brand distinction is not an injury-risk distinction.

Per player-week, the surface actually played on:

{_table(surf)}

Club home surfaces, which are a stadium property rather than a staff choice and the canonical
mechanism behind knee and ankle risk:

{_table(home_surf)}

Short weeks (four or five days' rest — Thursday games):

{_table(rest)}

## 4. Who is exposed

{_table(by_pos)}

## 5. Prior injury history — the covariate that cuts both ways

Mean prior designated weeks per risk row: **{hist["prior_designated_weeks"].mean():.1f}**
(median {hist["prior_designated_weeks"].median():.0f},
max {hist["prior_designated_weeks"].max():,}).

History is counted **strictly from prior seasons**, so a season never contributes to its own
covariate, and from the **injury report back to {FIRST_INJURY_SEASON}** rather than from
episodes — the report is comparable across that whole span even though the roster fields that
define episodes only become comparable at {FIRST_COMPARABLE_SEASON}.

**This covariate is not neutral, and stage 4 must not treat it as such.** A poor availability
system manufactures players who look fragile, so prior injuries are partly its own output.
Adjusting for them therefore adjusts away part of the effect being measured. Stage 4 fits
incidence **with and without** history and reports the pair as a **bound** — no-history as the
upper bound on the club effect, with-history as the lower — rather than picking one and calling
it the answer. That is why history ships as its own table instead of being folded in here.
"""
    ensure_dir(path.parent)
    path.write_text(md)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seasons", nargs=2, type=int, metavar=("START", "END"),
                        default=[FIRST_COMPARABLE_SEASON, LATEST_SEASON],
                        help=f"inclusive season range "
                             f"(default {FIRST_COMPARABLE_SEASON}-{LATEST_SEASON})")
    args = parser.parse_args(argv)

    seasons = tuple(
        s for s in range(args.seasons[0], args.seasons[1] + 1) if s not in EXCLUDED_SEASONS
    )
    result = build(seasons)

    ensure_dir(DATA_PROCESSED)
    p1 = write_parquet(result["exposure"], DATA_PROCESSED / "medstaff_exposure.parquet")
    p2 = write_parquet(result["history_group"], DATA_PROCESSED / "medstaff_history.parquet")
    p3 = write_report(result, REPORTS_DIR / "REPORT_medstaff_exposure.md", seasons=seasons)

    print(f"{result['exposure'].height:,} risk player-weeks ({seasons[0]}–{seasons[-1]})")
    for path in (p1, p2, p3):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
