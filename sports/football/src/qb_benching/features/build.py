"""Preseason features: everything about an opening starter knowable by **Sept 1** of his season.

The horizon is the whole design constraint. The model is meant to be read at a draft, before a
snap is played, so every column here comes from a completed prior season, the spring's draft, or
the club's own week-1 depth chart. Nothing is allowed to look at the season being predicted.

Four families, each an ``add_*(frame, ...) -> (frame, columns)`` block in the house style, with
explicit ``has_*`` flags rather than silent NaNs:

``prior_play``
    What he did last year — volume, efficiency, and how much he actually started.
``tenure``
    Who he is — age, experience, draft capital, how long he has been starting.
``room``
    Who is behind him, and what the club did about the position in the offseason.
``team``
    What the club did last year, and whether the man deciding is new.

⚠️ **`features/build.py::add_offseason` from the ranking pipeline is deliberately not reused.**
It computes almost exactly the ``room`` family, but it merges on ``team_next`` taken from the
*full-season* N+1 roster, and PROMPT_LOG entry 096 established that a missing ``team_next``
implies ``games_next == 0`` with probability 1.000 — a deterministic label leak for any
absence-shaped target, which is what this one is. Every column here is built from the **week-1**
roster and depth chart instead, which is a genuine Sept-1 view.
"""

from __future__ import annotations

from medstaff.data.teams import canonicalize

REG = "REG"


def _qb_season_totals(weekly):
    """Per ``(player_id, season)`` passing and rushing totals, from the weekly box scores."""
    import polars as pl

    w = weekly.filter((pl.col("season_type") == REG) & (pl.col("position") == "QB"))
    z = lambda c: pl.col(c).fill_null(0)  # noqa: E731
    return w.group_by(["player_id", "season"]).agg([
        z("attempts").sum().alias("attempts"),
        z("completions").sum().alias("completions"),
        z("passing_yards").sum().alias("passing_yards"),
        z("passing_tds").sum().alias("passing_tds"),
        z("interceptions").sum().alias("interceptions"),
        z("sacks").sum().alias("sacks"),
        z("passing_epa").sum().alias("passing_epa"),
        (z("passing_cpoe") * z("attempts")).sum().alias("_cpoe_w"),
        z("carries").sum().alias("carries"),
        z("rushing_yards").sum().alias("rushing_yards"),
        z("rushing_tds").sum().alias("rushing_tds"),
        pl.len().alias("appearances"),
    ])


def add_prior_play(cohort, weekly, starters):
    """Season *N-1* production for the opener. The single most obvious thing a coach knows.

    Rates are per dropback (attempts + sacks), which is the denominator that counts a sack as the
    play it was rather than discarding it. ``prior_starts`` comes from the announced-starter table
    rather than from appearances, so a mop-up week never reads as a start.
    """
    import polars as pl

    tot = _qb_season_totals(weekly).with_columns(
        (pl.col("attempts") + pl.col("sacks")).alias("dropbacks"))
    tot = tot.with_columns([
        (pl.col("passing_epa") / pl.col("dropbacks").clip(1)).alias("prior_epa_per_db"),
        (pl.col("_cpoe_w") / pl.col("attempts").clip(1)).alias("prior_cpoe"),
        (pl.col("completions") / pl.col("attempts").clip(1)).alias("prior_comp_pct"),
        (pl.col("passing_yards") / pl.col("attempts").clip(1)).alias("prior_ypa"),
        (pl.col("passing_tds") / pl.col("attempts").clip(1)).alias("prior_td_rate"),
        (pl.col("interceptions") / pl.col("attempts").clip(1)).alias("prior_int_rate"),
        (pl.col("sacks") / pl.col("dropbacks").clip(1)).alias("prior_sack_rate"),
        (pl.col("carries") / pl.col("dropbacks").clip(1)).alias("prior_rush_share"),
        pl.col("attempts").alias("prior_attempts"),
        pl.col("dropbacks").alias("prior_dropbacks"),
    ])
    starts = (
        starters.filter(pl.col("starter_id").is_not_null())
        .group_by(["starter_id", "season"]).agg(pl.len().alias("prior_starts"))
        .rename({"starter_id": "player_id"})
    )
    prior = tot.join(starts, on=["player_id", "season"], how="left").with_columns(
        pl.col("prior_starts").fill_null(0),
        (pl.col("season") + 1).cast(pl.Int32).alias("season"),
    ).rename({"player_id": "opener_id"})

    cols = ["prior_attempts", "prior_dropbacks", "prior_starts", "prior_epa_per_db",
            "prior_cpoe", "prior_comp_pct", "prior_ypa", "prior_td_rate", "prior_int_rate",
            "prior_sack_rate", "prior_rush_share"]
    out = cohort.join(prior.select(["opener_id", "season", *cols]), on=["opener_id", "season"],
                      how="left")
    out = out.with_columns(pl.col("prior_attempts").is_not_null().alias("has_prior_season"))
    # A quarterback with no prior season is a rookie or a returnee, not a zero-volume starter;
    # volume is filled at zero because "did not play" is the fact, while rates stay null.
    out = out.with_columns([pl.col(c).fill_null(0) for c in
                            ("prior_attempts", "prior_dropbacks", "prior_starts")])
    return out, [*cols, "has_prior_season"]


def add_tenure(cohort, rosters, draft_picks):
    """Who he is: age, experience, draft capital, and how established the job is."""
    import polars as pl

    r = (rosters.select([
            pl.col("season").cast(pl.Int32),
            pl.col("player_id").alias("opener_id"),
            pl.col("years_exp").cast(pl.Float64, strict=False).alias("years_exp"),
            pl.col("birth_date"),
         ])
         .unique(subset=["season", "opener_id"], keep="first"))
    out = cohort.join(r, on=["opener_id", "season"], how="left")
    out = out.with_columns(
        (pl.col("season") - pl.col("birth_date").cast(pl.Date, strict=False).dt.year())
        .cast(pl.Float64).alias("age")
    ).drop("birth_date")

    d = (draft_picks.select([
            pl.col("gsis_id").alias("opener_id"),
            pl.col("pick").cast(pl.Float64, strict=False).alias("draft_pick"),
            pl.col("round").cast(pl.Float64, strict=False).alias("draft_round"),
         ]).drop_nulls("opener_id").unique(subset="opener_id", keep="first"))
    out = out.join(d, on="opener_id", how="left")
    out = out.with_columns([
        pl.col("draft_pick").is_not_null().alias("was_drafted"),
        (pl.col("years_exp") == 0).alias("is_rookie"),
    ])
    # how many seasons he has already opened, before this one
    prior_openers = (
        cohort.select(["opener_id", "season"]).sort("season")
        .group_by("opener_id").agg(pl.col("season").alias("_seasons"))
    )
    out = out.join(prior_openers, on="opener_id", how="left")
    out = out.with_columns(
        pl.struct(["season", "_seasons"]).map_elements(
            lambda s: sum(1 for x in (s["_seasons"] or []) if x < s["season"]),
            return_dtype=pl.Int32).alias("prior_opening_seasons")
    ).drop("_seasons")
    return out, ["years_exp", "age", "draft_pick", "draft_round", "was_drafted", "is_rookie",
                 "prior_opening_seasons"]


def add_room(cohort, depth, weekly, draft_picks, starters):
    """The quarterback room as it stood in week 1 — the competition, not the depth chart's owner.

    Everything is taken from the **week-1** depth chart. A benching needs somebody to bench *to*,
    so the quality of the man at rank 2 is the most direct competitive pressure there is.

    ``incumbent_present`` is the flag the plan doc asks for: last season's primary starter for
    this club is on the week-1 chart but is *not* the opener. That is the Derek Anderson shape —
    an opener who has the job only until somebody gets healthy — and losing it back is not a
    benching on the merits, so the model must be able to see the difference.
    """
    import polars as pl

    wk1 = depth.filter(pl.col("week") == 1)
    room = wk1.group_by(["season", "team"]).agg(pl.len().alias("room_size"))

    tot = _qb_season_totals(weekly).with_columns(
        (pl.col("attempts") + pl.col("sacks")).alias("dropbacks"))
    prior = tot.select([
        pl.col("player_id"),
        (pl.col("season") + 1).cast(pl.Int32).alias("season"),
        pl.col("attempts").alias("_att"),
        (pl.col("passing_epa") / pl.col("dropbacks").clip(1)).alias("_epa"),
    ])

    # the backup: rank 2 on the week-1 chart, with his own prior-season production
    backup = (wk1.filter(pl.col("depth_rank") == 2)
                 .select(["season", "team", pl.col("opener_id").alias("player_id")])
                 .unique(subset=["season", "team"], keep="first")
                 .join(prior, on=["player_id", "season"], how="left"))
    backup = backup.select([
        "season", "team",
        pl.col("_att").fill_null(0).alias("backup_prior_attempts"),
        pl.col("_epa").alias("backup_prior_epa_per_db"),
    ])

    # a quarterback the club drafted this spring, and how high
    rookie = (draft_picks.filter(pl.col("position") == "QB")
              .select([pl.col("season").cast(pl.Int32),
                       pl.col("team"),
                       pl.col("pick").cast(pl.Float64, strict=False).alias("_pick")])
              .group_by(["season", "team"]).agg(pl.col("_pick").min().alias("rookie_qb_pick")))

    # last season's primary starter for this club
    incumbent = (starters.filter(pl.col("starter_id").is_not_null())
                 .group_by(["season", "team", "starter_id"]).agg(pl.len().alias("n"))
                 .sort("n", descending=True)
                 .group_by(["season", "team"]).first()
                 .select(["season", "team", pl.col("starter_id").alias("_incumbent")])
                 .with_columns((pl.col("season") + 1).cast(pl.Int32).alias("season")))
    on_chart = wk1.select(["season", "team", pl.col("opener_id").alias("_incumbent")]).with_columns(
        _on_chart=True)
    incumbent = incumbent.join(on_chart, on=["season", "team", "_incumbent"], how="left")

    out = (cohort.join(room, on=["season", "team"], how="left")
                 .join(backup, on=["season", "team"], how="left")
                 .join(rookie, on=["season", "team"], how="left")
                 .join(incumbent, on=["season", "team"], how="left"))
    out = out.with_columns([
        pl.col("rookie_qb_pick").is_not_null().alias("rookie_qb_drafted"),
        pl.col("backup_prior_attempts").fill_null(0),
        (pl.col("_on_chart").fill_null(False) & (pl.col("_incumbent") != pl.col("opener_id")))
        .alias("incumbent_present"),
    ]).drop(["_incumbent", "_on_chart"])
    return out, ["room_size", "backup_prior_attempts", "backup_prior_epa_per_db",
                 "rookie_qb_pick", "rookie_qb_drafted", "incumbent_present"]


def add_team(cohort, schedules, head_coach):
    """What the club did last year, and whether the man who decides is new.

    A new head coach has no investment in the incumbent, which is the standard story for a
    quarterback losing a job he had held. Prior record and point differential stand in for how
    much patience the situation has.
    """
    import polars as pl

    reg = schedules.filter(pl.col("game_type") == REG)
    sides = []
    for side, other in (("home", "away"), ("away", "home")):
        sides.append(reg.select([
            pl.col("season").cast(pl.Int32),
            pl.col(f"{side}_team").alias("team"),
            pl.col(f"{side}_score").cast(pl.Float64).alias("pf"),
            pl.col(f"{other}_score").cast(pl.Float64).alias("pa"),
        ]))
    # Canonicalise: the cohort's team codes are canonical, so a relocated franchise (STL/SD/OAK)
    # would otherwise fail the join and silently lose its prior season.
    games = canonicalize(pl.concat(sides).drop_nulls(["pf", "pa"]), "team")
    rec = games.group_by(["season", "team"]).agg([
        (pl.col("pf") > pl.col("pa")).mean().alias("prior_win_pct"),
        (pl.col("pf") - pl.col("pa")).mean().alias("prior_point_diff"),
    ]).with_columns((pl.col("season") + 1).cast(pl.Int32).alias("season"))

    hc = head_coach.select([pl.col("season").cast(pl.Int32), "team", "coach"])
    prev = hc.with_columns((pl.col("season") + 1).cast(pl.Int32).alias("season"),
                           pl.col("coach").alias("_prev_coach")).drop("coach")
    out = (cohort.join(rec, on=["season", "team"], how="left")
                 .join(hc, on=["season", "team"], how="left")
                 .join(prev, on=["season", "team"], how="left"))
    out = out.with_columns(
        (pl.col("_prev_coach").is_not_null() & (pl.col("coach") != pl.col("_prev_coach")))
        .alias("new_head_coach")
    ).drop(["coach", "_prev_coach"])
    return out, ["prior_win_pct", "prior_point_diff", "new_head_coach"]


def add_history(cohort):
    """What happened to him the *last* time he opened a season.

    The lagged label. Known by Sept 1 by definition — it is last season's outcome — and the most
    obvious candidate for the strongest single feature, which is also why it needs a leakage test
    of its own: it must read season N-1 and never season N.
    """
    import polars as pl

    prev = cohort.select([
        pl.col("opener_id"),
        (pl.col("season") + 1).cast(pl.Int32).alias("season"),
        pl.col("benched").alias("benched_last_season"),
        pl.col("displaced").alias("displaced_last_season"),
        pl.col("n_benched").alias("n_benched_last_season"),
    ])
    out = cohort.join(prev, on=["opener_id", "season"], how="left")
    out = out.with_columns([
        pl.col("benched_last_season").is_not_null().alias("opened_last_season"),
        pl.col("benched_last_season").fill_null(False),
        pl.col("displaced_last_season").fill_null(False),
        pl.col("n_benched_last_season").fill_null(0),
    ])
    return out, ["benched_last_season", "displaced_last_season", "n_benched_last_season",
                 "opened_last_season"]


def build_features(cohort, *, weekly, starters, rosters, draft_picks, depth, schedules,
                   head_coach):
    """Assemble every block and return ``(frame, block_columns)``.

    The block map is written out beside the parquet, the same way the ranking pipeline does it,
    so the model layer can name feature groups rather than hard-coding column lists.
    """
    blocks = {}
    frame = cohort
    frame, blocks["prior_play"] = add_prior_play(frame, weekly, starters)
    frame, blocks["tenure"] = add_tenure(frame, rosters, draft_picks)
    frame, blocks["room"] = add_room(frame, depth, weekly, draft_picks, starters)
    frame, blocks["team"] = add_team(frame, schedules, head_coach)
    frame, blocks["history"] = add_history(frame)
    return frame, blocks


def feature_columns(blocks, *, exclude=()):
    """Flatten the block map into a column list, dropping named blocks."""
    return [c for name, cols in blocks.items() if name not in exclude for c in cols]
