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

# Injuries exist from 2009, but snap counts — the report-independent anchor the design leans
# on — only from 2012, so the modelling sample starts there. Earlier seasons are still loaded:
# they populate each player's prior-injury-history lookback, where missing snaps do not matter.
FIRST_INJURY_SEASON = 2009
FIRST_MODEL_SEASON = 2012
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
    return frame.with_columns(
        abbr.str.starts_with(RESERVE_PREFIX).alias("on_reserve"),
        abbr.str.starts_with(PRACTICE_SQUAD_PREFIX).alias("on_practice_squad"),
        (pl.col("status") == "INA").alias("inactive"),
        (pl.col("status") == "ACT").alias("active"),
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
    "ACTIVE_CODE", "EXCLUDED_SEASONS", "FIRST_INJURY_SEASON", "FIRST_MODEL_SEASON",
    "PRACTICE_SQUAD_PREFIX", "RAW_DATASETS", "RESERVE_PREFIX",
    "body_part_summary", "join_rate", "listing_propensity", "load_injuries", "load_raw",
    "load_rosters_weekly", "load_schedules", "report_regime_table", "season_coverage",
]
