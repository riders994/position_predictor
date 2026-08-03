"""Exposure and confounders: everything a club's medical staff does *not* control.

Stage 4 measures observed minus expected, so this stage assembles the "expected" side's inputs.
The covariates here are deliberately the ones outside the training room's influence — how old the
roster is, how much football each player is asked to play, what surface he plays it on, how much
rest he gets. Anything the staff plausibly *does* control is excluded on purpose; putting it here
would adjust away the effect the project exists to measure.

One covariate is deliberately ambiguous and gets special handling downstream: **prior injury
history**. A poor availability system manufactures players who look fragile, so history is partly
its own output. Stage 4 fits incidence with and without it and reports the pair as a bound rather
than a point — which is why :func:`injury_history` is a separate table rather than folded in.

**The prior-injury lookback reaches back to 2009 even though episodes start in 2021.** The injury
*report* (body part, practice participation) is comparable across that whole span; it is the
*roster* fields that break at 2021 (see ``data.ingest.FIRST_COMPARABLE_SEASON``). So history is
counted in designated weeks from the report, not in episodes.
"""

from __future__ import annotations

# Surface strings are free text and dirty: "grass " with a trailing space is a separate value
# from "grass" (93 vs 612 rows in 2021-2025), and "" is used for missing. Turf brands are
# distinctions without a difference for injury risk, so they collapse to one level.
TURF_TOKENS = ("turf", "astro", "sport", "matrix", "field", "a_turf")
GRASS_TOKENS = ("grass",)

INDOOR_ROOFS = ("dome", "closed")

# nflverse carries a handful of roster rows for players who have not played in decades (one 2023
# row has years_exp=29). Clip rather than drop: the row is real exposure, the number is not.
MAX_YEARS_EXP = 22


def normalize_surface(raw: str | None) -> str | None:
    """Collapse the free-text surface field to ``grass`` / ``turf`` / ``None``."""
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if not text:
        return None
    if any(t in text for t in GRASS_TOKENS):
        return "grass"
    if any(t in text for t in TURF_TOKENS):
        return "turf"
    return None


def game_context(schedules):
    """Per (season, week, team): the conditions that club played in that week."""
    import polars as pl

    reg = schedules.filter(pl.col("game_type") == "REG")
    sides = []
    for side, opp in (("home", "away"), ("away", "home")):
        sides.append(reg.select([
            "season", "week",
            pl.col(f"{side}_team").alias("team"),
            pl.col(f"{opp}_team").alias("opponent"),
            pl.col(f"{side}_rest").alias("rest_days"),
            pl.lit(side == "home").alias("is_home"),
            pl.col("surface"), pl.col("roof"), pl.col("temp"), pl.col("div_game"),
        ]))
    out = pl.concat(sides)
    return out.with_columns(
        pl.col("surface").map_elements(normalize_surface, return_dtype=pl.String)
        .alias("surface"),
        pl.col("roof").is_in(INDOOR_ROOFS).alias("indoor"),
        (pl.col("rest_days") <= 5).alias("short_week"),
        pl.lit(True).alias("team_played"),
    ).drop("roof")


def club_home_surface(schedules):
    """Per (season, team): the surface of the club's own stadium.

    A club property, not a staff choice — and the canonical mechanism behind knee and ankle
    risk, which is why the signature analysis in stage 5 wants it separately from the per-game
    surface a player was actually exposed to.
    """
    import polars as pl

    reg = schedules.filter(pl.col("game_type") == "REG")
    return (
        reg.select([
            "season", pl.col("home_team").alias("team"),
            pl.col("surface").map_elements(normalize_surface, return_dtype=pl.String)
            .alias("home_surface"),
        ])
        .drop_nulls("home_surface")
        .group_by(["season", "team"])
        .agg(pl.col("home_surface").mode().first())
    )


def player_attributes(rosters_weekly):
    """Per (season, gsis_id): age at 1 September, experience, size."""
    import polars as pl

    one = rosters_weekly.sort(["season", "gsis_id", "week"]).unique(
        subset=["season", "gsis_id"], keep="first", maintain_order=True
    )
    # birth_date must be cast explicitly: a column that is entirely null arrives typed Null,
    # and subtracting it from a Date raises rather than yielding a null age.
    one = one.with_columns(pl.col("birth_date").cast(pl.Date, strict=False))
    season_start = pl.date(pl.col("season"), 9, 1)
    return one.select([
        "season", "gsis_id", "position", "position_group", "side",
        ((season_start - pl.col("birth_date")).dt.total_days() / 365.25).alias("age"),
        pl.col("years_exp").clip(0, MAX_YEARS_EXP).alias("years_exp"),
        pl.col("height").alias("height_in"),
        pl.col("weight").alias("weight_lb"),
        (703.0 * pl.col("weight") / (pl.col("height") ** 2)).alias("bmi"),
    ]).with_columns((pl.col("years_exp") == 0).alias("rookie"))


def snap_exposure(snap_counts, crosswalk):
    """Per (season, week, gsis_id): snaps played, and trailing workload.

    Snaps are an *intensity* covariate here, never an availability signal — that is the roster's
    job (see ``episodes.build``). The crosswalk is lossy, so every column is null-safe and the
    match rate is reported rather than assumed.
    """
    import polars as pl

    sc = snap_counts.filter(pl.col("game_type") == "REG").with_columns(
        pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32)
    )
    joined = (
        sc.join(crosswalk, left_on="pfr_player_id", right_on="pfr_id", how="left")
        .filter(pl.col("gsis_id").is_not_null())
        .with_columns(
            (pl.col("offense_snaps").fill_null(0) + pl.col("defense_snaps").fill_null(0))
            .alias("snaps"),
            pl.max_horizontal(
                pl.col("offense_pct").fill_null(0), pl.col("defense_pct").fill_null(0)
            ).alias("snap_share"),
        )
        .group_by(["season", "week", "gsis_id"])
        .agg(pl.col("snaps").sum(), pl.col("snap_share").max())
        .sort(["gsis_id", "season", "week"])
    )
    return joined.with_columns(
        pl.col("snaps").rolling_mean(3, min_periods=1)
        .over(["gsis_id", "season"]).alias("snaps_3wk"),
        pl.col("snaps").cum_sum().over(["gsis_id", "season"]).alias("snaps_cum"),
        pl.col("snap_share").rolling_mean(3, min_periods=1)
        .over(["gsis_id", "season"]).alias("snap_share_3wk"),
    )


def id_crosswalk(players, ids=None, rosters_weekly=None):
    """pfr_player_id -> gsis_id, preferring the source whose coverage is not position-biased.

    ``players`` is the only complete route. The obvious alternatives are both biased in the one
    direction that would corrupt this project:

    - ``rosters_weekly.pfr_id`` is null for **99.8% of offensive-line rows** (against ~10-24%
      elsewhere)
    - ``ids`` is a *fantasy* table: of 352 distinct linemen in a single season's snap counts, it
      resolves **two**

    Either would leave snap-workload covariates present for skill players and absent for
    linemen — and position correlates with body part, which is exactly what the stage-5
    signature analysis compares. ``players`` is ~12% null for linemen and ~11% for skill
    players, i.e. missing at roughly the same rate everywhere.
    """
    import polars as pl

    parts = []
    for source in (players, ids, rosters_weekly):
        if source is not None and {"gsis_id", "pfr_id"} <= set(source.columns):
            parts.append(source.select(["gsis_id", "pfr_id"]))
    if not parts:
        raise KeyError("no source carries both gsis_id and pfr_id")
    return pl.concat(parts).drop_nulls().unique(subset=["pfr_id"], keep="first")


def injury_history(injuries, *, seasons):
    """Per (season, gsis_id): designated weeks in *prior* seasons, overall and by body group.

    Strictly prior — a season never contributes to its own history, or the covariate would leak
    the outcome. Counted from the injury report, which is comparable back to 2009, rather than
    from episodes, which only exist from 2021.
    """
    import polars as pl

    inj = injuries.filter(
        (pl.col("game_type") == "REG")
        & pl.col("body_part_group").is_not_null()
        & ~pl.col("body_part_group").is_in(["non_injury", "illness"])
    )
    per_season = (
        inj.select(["season", "gsis_id", "body_part_group", "week"]).unique()
        .group_by(["season", "gsis_id", "body_part_group"]).len()
        .rename({"len": "weeks"})
    )

    rows = []
    for target in seasons:
        prior = per_season.filter(pl.col("season") < target)
        if prior.is_empty():
            continue
        overall = prior.group_by("gsis_id").agg(
            pl.col("weeks").sum().alias("prior_designated_weeks"),
            pl.col("season").n_unique().alias("prior_injury_seasons"),
        ).with_columns(pl.lit(target).alias("season"))
        rows.append(overall)
    overall = pl.concat(rows) if rows else pl.DataFrame(
        schema={"gsis_id": pl.String, "prior_designated_weeks": pl.UInt32,
                "prior_injury_seasons": pl.UInt32, "season": pl.Int32})

    group_rows = []
    for target in seasons:
        prior = per_season.filter(pl.col("season") < target)
        if prior.is_empty():
            continue
        group_rows.append(
            prior.group_by(["gsis_id", "body_part_group"]).agg(
                pl.col("weeks").sum().alias("prior_group_weeks")
            ).with_columns(pl.lit(target).alias("season"))
        )
    by_group = pl.concat(group_rows) if group_rows else pl.DataFrame(
        schema={"gsis_id": pl.String, "body_part_group": pl.String,
                "prior_group_weeks": pl.UInt32, "season": pl.Int32})

    return overall, by_group


def build_risk_set(panel, episodes):
    """Player-weeks eligible to *start* an episode, with weeks-since-return attached.

    A week already inside an open spell is not at risk of a new onset — counting it would make a
    long absence look like many weeks of injury-free exposure. Practice-squad weeks and byes are
    excluded too: neither is exposure to an NFL game.
    """
    import polars as pl

    covered = []
    if not episodes.is_empty():
        for ep in episodes.select(
            ["season", "gsis_id", "onset_week", "end_week", "return_week"]
        ).iter_rows(named=True):
            for w in range(ep["onset_week"], (ep["end_week"] or ep["onset_week"]) + 1):
                covered.append((ep["season"], ep["gsis_id"], w))
    covered_df = pl.DataFrame(
        covered, schema=["season", "gsis_id", "week"], orient="row"
    ).unique().with_columns(pl.lit(True).alias("in_episode")) if covered else None

    risk = panel.filter(
        pl.col("team_played") & ~pl.col("on_practice_squad")
    )
    if covered_df is not None:
        risk = risk.join(covered_df, on=["season", "gsis_id", "week"], how="left").filter(
            pl.col("in_episode").is_null()
        ).drop("in_episode")

    # Weeks since the most recent return, within the season — a documented reinjury risk factor
    # and, unlike prior history, unambiguously not the staff's own output.
    if not episodes.is_empty():
        returns = (
            episodes.filter(pl.col("return_week").is_not_null())
            .select(["season", "gsis_id", pl.col("return_week").alias("week")])
            .with_columns(pl.lit(True).alias("returned"))
        )
        risk = risk.join(returns, on=["season", "gsis_id", "week"], how="left")
        risk = risk.sort(["gsis_id", "season", "week"]).with_columns(
            pl.when(pl.col("returned").fill_null(False)).then(pl.col("week"))
            .otherwise(None).forward_fill().over(["gsis_id", "season"]).alias("_last_return")
        ).with_columns(
            (pl.col("week") - pl.col("_last_return")).alias("weeks_since_return")
        ).drop("returned", "_last_return")
    return risk


__all__ = [
    "GRASS_TOKENS", "INDOOR_ROOFS", "MAX_YEARS_EXP", "TURF_TOKENS",
    "build_risk_set", "club_home_surface", "game_context", "id_crosswalk", "injury_history",
    "normalize_surface", "player_attributes", "snap_exposure",
]
