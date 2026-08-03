"""Load and validate the three raw datasets this project adds, and prove they join.

Stage 1 exists to make the data's defects visible *before* anything is modelled on it. Three of
them are load-bearing for the rest of the project and each gets an explicit diagnostic here:

1. **The 2016 regime break.** ``report_status`` is 3-6% null for 2009-2015 and 39-55% null from
   2016, when the league dropped the "Probable" designation. Anything built on that column is
   not comparable across the boundary. :func:`report_regime_table` measures it per season so the
   choice to build on the regime-invariant columns instead is evidenced, not asserted.
2. **Disclosure behaviour varies by club.** :func:`listing_propensity` computes the three
   indices used later to separate reporting culture from medicine.
3. **The spine has to actually join.** :func:`join_rate` checks injuries against weekly rosters;
   a silent loss here would read downstream as absence.
"""

from __future__ import annotations

from pathlib import Path

from .positions import attach_position_group
from .taxonomy import classify, coalesce_body_part
from .teams import canonicalize

# Injuries exist from 2009. The injury *report* (body part, practice status) is comparable from
# 2009 onward and supplies the prior-injury-history lookback at any depth.
FIRST_INJURY_SEASON = 2009

# ⚠️ Absence measures are only comparable from 2021. `rosters_weekly` changes what it records
# at that boundary: the gameday active/inactive split is absent earlier (INA is 2.8k rows across
# 2012-2019 vs 16.8k across 2021-2025; ACT falls 0.86 -> 0.59), and the reserve codes in
# `status_description_abbr` are unpopulated before 2021. Anything counting missed games — every
# episode, duration and recurrence outcome — is therefore restricted to this window. It happens
# to be exactly the two grading windows (5yr 2021-2025, 3yr 2023-2025), the post-COVID 17-game
# era and the post-2016 reporting regime, so the sample is one homogeneous regime rather than a
# compromise. See :func:`roster_status_regime_table`.
FIRST_COMPARABLE_SEASON = 2021

# Reserve/injured lists as spelled in `status`. Stable at 6-14% of rows in every season since
# 2012, unlike the abbr codes.
RESERVE_STATUSES = ("RES", "PUP", "NON", "EXE")

# ⚠️ Reserve/COVID-19. `R59` occurs in **2021 only** (725 player-weeks) and is exactly zero in
# every other season — it is the pandemic reserve list, which carries `RES` status without being
# an injury. Left in, it inflates 2021's reserve share to 0.46 against ~0.31 for 2022-2025 and
# makes every club look worse at medicine in the first year of the five-year window.
COVID_RESERVE_CODES = ("R59",)
# 2020's practice and roster regime was unlike any other season; the sibling pipeline already
# excludes it, and this project follows so the windows stay comparable.
EXCLUDED_SEASONS = (2020,)

RAW_DATASETS = ("injuries", "rosters_weekly", "schedules")

# Reserve/injury status codes. R01 is Injured Reserve, R02 IR-designated-to-return; the rest of
# the R-family are PUP, NFI and similar. Prefix-matching rather than an enumerated set, because
# nflverse adds codes and an unrecognised R-code silently reading as "active" is the dangerous
# direction of that error.
RESERVE_PREFIX = "R"
PRACTICE_SQUAD_PREFIX = "P"
ACTIVE_CODE = "A01"


def _normalize_keys(frame):
    """Cast the join keys to a single dtype across datasets.

    The caches disagree: ``rosters_weekly`` lands ``season``/``week`` as f64 (nflreadpy
    concatenates its multi-season pulls with ``diagonal_relaxed``, which widens ints once any
    season has a null), while ``injuries`` keeps i32. Polars refuses to join across that, and a
    join key silently coerced the other way would be worse — so both are pinned here.
    """
    import polars as pl

    casts = [pl.col(c).cast(pl.Int32, strict=False).alias(c)
             for c in ("season", "week") if c in frame.columns]
    return frame.with_columns(casts) if casts else frame


def _require(path: Path, dataset: str):
    if not path.exists():
        raise SystemExit(
            f"missing {path}\nFetch the raw caches first:\n"
            f"  make -C sports/football fetch DATASETS={dataset}"
        )
    return path


def load_raw(data_raw: Path, name: str):
    """Read one cached raw dataset as polars, failing with the command that produces it."""
    import polars as pl

    return _normalize_keys(pl.read_parquet(_require(data_raw / f"{name}.parquet", name)))


def load_injuries(data_raw: Path, *, seasons: tuple[int, ...] | None = None):
    """Injuries with the body part coalesced, classified, and teams/positions canonicalised."""
    import polars as pl

    frame = load_raw(data_raw, "injuries")
    if seasons is not None:
        frame = frame.filter(pl.col("season").is_in(list(seasons)))
    frame = canonicalize(frame, "team")
    frame = coalesce_body_part(frame)
    frame = classify(frame)
    return attach_position_group(frame)


def load_rosters_weekly(data_raw: Path, *, seasons: tuple[int, ...] | None = None):
    """Weekly rosters — the availability spine — with status flags derived."""
    import polars as pl

    frame = load_raw(data_raw, "rosters_weekly")
    if seasons is not None:
        frame = frame.filter(pl.col("season").is_in(list(seasons)))
    frame = canonicalize(frame, "team")
    frame = attach_position_group(frame)
    abbr = pl.col("status_description_abbr").fill_null("")
    # Reserve comes from `status`, NOT `status_description_abbr` — see FIRST_COMPARABLE_SEASON.
    # The abbr R-codes are 0.0% of rows for 2012-2015 and 51% null in 2016, so keying off them
    # makes IR look like a 2021 invention. `status == RES` is 6-14% in every season.
    return frame.with_columns(
        (pl.col("status").is_in(RESERVE_STATUSES)
         & ~abbr.is_in(COVID_RESERVE_CODES)).alias("on_reserve"),
        (abbr.str.starts_with(PRACTICE_SQUAD_PREFIX) | (pl.col("status") == "DEV"))
        .alias("on_practice_squad"),
        (pl.col("status") == "INA").alias("inactive"),
        (pl.col("status") == "ACT").alias("active"),
        abbr.str.starts_with(RESERVE_PREFIX).alias("reserve_code_present"),
    )


def roster_status_regime_table(rosters_weekly):
    """Per-season population of the roster-status fields — the comparability guard.

    Two fields silently change meaning at 2021 and both would have corrupted every absence
    measure in the project:

    ``status_description_abbr`` R-codes are **0.0%** of rows for 2012-2015 and 51% null in 2016,
    so reading IR off them makes injured reserve look like something clubs started doing in 2021.

    ``status == "INA"`` is barely populated before 2021 (2.8k rows across 2012-2019 against
    16.8k across 2021-2025) while ``ACT`` falls from 0.86 to 0.59 — the gameday active/inactive
    split is simply not in the earlier data, so games-missed is not comparable across the break.

    This is the same shape of defect the sibling project hit with cfbfastR play flags: an
    unpopulated field aggregates to a clean zero rather than a null, so nothing errors and the
    series just quietly means something different on each side of the boundary.
    """
    import polars as pl

    return (
        rosters_weekly.group_by("season")
        .agg(
            pl.len().alias("rows"),
            pl.col("on_reserve").mean().alias("reserve_share"),
            pl.col("reserve_code_present").mean().alias("abbr_R_share"),
            pl.col("active").mean().alias("active_share"),
            pl.col("inactive").mean().alias("inactive_share"),
        )
        .with_columns((pl.col("season") >= FIRST_COMPARABLE_SEASON).alias("comparable"))
        .sort("season")
    )


def load_schedules(data_raw: Path, *, seasons: tuple[int, ...] | None = None):
    """Schedules, canonicalised — supplies surface/roof confounders and head coaches."""
    import polars as pl

    frame = load_raw(data_raw, "schedules")
    if seasons is not None:
        frame = frame.filter(pl.col("season").is_in(list(seasons)))
    return canonicalize(frame, "home_team", "away_team")


def season_coverage(frame, *, label: str):
    """Rows, distinct players and distinct teams per season."""
    import polars as pl

    player = "gsis_id" if "gsis_id" in frame.columns else None
    aggs = [pl.len().alias("rows")]
    if player:
        aggs.append(pl.col(player).n_unique().alias("players"))
    if "team" in frame.columns:
        aggs.append(pl.col("team").n_unique().alias("teams"))
    return (
        frame.group_by("season").agg(aggs).sort("season")
        .with_columns(pl.lit(label).alias("dataset"))
    )


def report_regime_table(injuries):
    """Per-season null rates that expose the 2016 ``report_status`` break.

    The point of the table is the contrast: ``report_status`` breaks, ``practice_status`` and
    the coalesced body part do not. That contrast is the whole justification for building the
    project's outcomes on the latter two.
    """
    import polars as pl

    return (
        injuries.group_by("season")
        .agg(
            pl.len().alias("rows"),
            pl.col("report_status").null_count().alias("_rs"),
            pl.col("practice_status").null_count().alias("_ps"),
            pl.col("body_part_raw").null_count().alias("_bp"),
        )
        .with_columns(
            (pl.col("_rs") / pl.col("rows")).alias("report_status_null"),
            (pl.col("_ps") / pl.col("rows")).alias("practice_status_null"),
            (pl.col("_bp") / pl.col("rows")).alias("body_part_null"),
            (pl.col("season") >= 2016).alias("post_probable_drop"),
        )
        .drop("_rs", "_ps", "_bp")
        .sort("season")
    )


def listing_propensity(injuries, rosters_weekly):
    """Per team-season disclosure indices.

    ``listing_rate``            report player-weeks per active rostered player-week
    ``questionable_share``      Questionable / (Out + Doubtful + Questionable)
    ``questionable_play_rate``  share of Questionable-listed players who were not inactive —
                                the cleanest behavioural index, since a club that lists
                                everybody has a high play-rate among its Questionables

    These are covariates, not grades: a club that discloses more is not a club that injures
    more, and the whole point of computing them is to keep those two apart.
    """
    import polars as pl

    listed = injuries.group_by(["season", "team"]).agg(pl.len().alias("listed_weeks"))
    active = (
        rosters_weekly.filter(pl.col("active"))
        .group_by(["season", "team"]).agg(pl.len().alias("active_weeks"))
    )
    designated = injuries.filter(pl.col("report_status").is_in(["Out", "Doubtful", "Questionable"]))
    status = (
        designated.group_by(["season", "team"])
        .agg(
            pl.len().alias("designated"),
            (pl.col("report_status") == "Questionable").sum().alias("questionable"),
        )
    )
    played = (
        designated.filter(pl.col("report_status") == "Questionable")
        .join(
            rosters_weekly.select(["season", "week", "gsis_id", "inactive", "on_reserve"]).unique(
                subset=["season", "week", "gsis_id"]
            ),
            on=["season", "week", "gsis_id"], how="left",
        )
        .group_by(["season", "team"])
        .agg(
            (~(pl.col("inactive").fill_null(False) | pl.col("on_reserve").fill_null(False)))
            .sum().alias("questionable_available"),
            pl.len().alias("questionable_rows"),
        )
    )
    return (
        listed.join(active, on=["season", "team"], how="left")
        .join(status, on=["season", "team"], how="left")
        .join(played, on=["season", "team"], how="left")
        .with_columns(
            (pl.col("listed_weeks") / pl.col("active_weeks")).alias("listing_rate"),
            (pl.col("questionable") / pl.col("designated")).alias("questionable_share"),
            (pl.col("questionable_available") / pl.col("questionable_rows"))
            .alias("questionable_play_rate"),
        )
        .sort(["season", "team"])
    )


def join_rate(injuries, rosters_weekly):
    """Share of injury rows that find a weekly roster row on ``(season, week, gsis_id)``.

    Verified at 100% for 2023 during design. A drop here is the single most dangerous silent
    failure in the project: an injury row with no roster row has no availability spine, so the
    player's absence becomes invisible rather than obviously missing.
    """
    import polars as pl

    spine = rosters_weekly.select(["season", "week", "gsis_id", "status"]).unique(
        subset=["season", "week", "gsis_id"]
    )
    joined = injuries.join(spine, on=["season", "week", "gsis_id"], how="left")
    matched = joined.filter(pl.col("status").is_not_null()).height
    per_season = (
        joined.group_by("season")
        .agg(
            pl.len().alias("rows"),
            pl.col("status").is_not_null().sum().alias("matched"),
        )
        .with_columns((pl.col("matched") / pl.col("rows")).alias("match_rate"))
        .sort("season")
    )
    return {"rows": joined.height, "matched": matched,
            "rate": matched / joined.height if joined.height else float("nan"),
            "per_season": per_season}


def body_part_summary(injuries):
    """Episode-agnostic body-part counts, focal parts flagged."""
    import polars as pl

    from .taxonomy import FOCAL_GROUPS

    return (
        injuries.filter(pl.col("body_part_group").is_not_null())
        .group_by("body_part_group").len().rename({"len": "rows"})
        .with_columns(
            (pl.col("rows") / injuries.height).alias("share"),
            pl.col("body_part_group").is_in(list(FOCAL_GROUPS)).alias("focal"),
        )
        .sort("rows", descending=True)
    )


__all__ = [
    "ACTIVE_CODE", "EXCLUDED_SEASONS", "FIRST_COMPARABLE_SEASON", "FIRST_INJURY_SEASON",
    "PRACTICE_SQUAD_PREFIX", "RAW_DATASETS", "RESERVE_PREFIX",
    "RESERVE_STATUSES", "COVID_RESERVE_CODES", "roster_status_regime_table",
    "body_part_summary", "join_rate", "listing_propensity", "load_injuries", "load_raw",
    "load_rosters_weekly", "load_schedules", "report_regime_table", "season_coverage",
]
