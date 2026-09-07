"""The club's own declared quarterback ordering, week by week.

This is the project's primary role signal, and the reason it exists is a defect found in the
alternatives. ``rosters_weekly.status`` cannot carry a weekly role before 2021 (it is a
season-final stamp — see :mod:`qb_benching.labels.displacement`), and the injury report cannot
see injured reserve at all. The depth chart has neither problem: it is published weekly from
2001, and it records the two things the label needs **directly** rather than by residual.

Colin Kaepernick's 2015, the case that exposed the roster defect, reads straight off it:

===========  ====================================
weeks 1-9    Kaepernick ``depth_team`` 1
week 10      Kaepernick 2, **Gabbert 1** — the benching, as the club declared it
weeks 11+    Kaepernick absent from the chart — he had gone on injured reserve
===========  ====================================

⚠️ Two schemas
--------------
nflverse changed the feed for 2025. Through 2024 it is a weekly table keyed on
``(season, week, club_code)`` with a string ``depth_team`` rank. From 2025 it is a **dated
snapshot** feed — ``dt``, ``team``, ``pos_abb``, integer ``pos_rank`` — with no season or week
column at all, and roughly 15x the rows because it is published every few days rather than
weekly. :func:`qb_depth` normalises both to one frame and, for the snapshot era, resolves each
team-game to the **latest snapshot strictly before kickoff**, which is the chart as it stood when
the club named its starter.

A useful side effect: the snapshot feed starts in early August, so it also carries the preseason
chart a live board needs in order to know who is opening the season.
"""

from __future__ import annotations

from medstaff.data.teams import canonicalize

#: Rank 1 on the chart — the club's declared starter.
STARTER_RANK = 1


def _legacy(depth_charts):
    """The weekly ``(season, week, club_code, depth_team)`` schema, 2001-2024."""
    import polars as pl

    d = depth_charts.filter(
        pl.col("season").is_not_null() & (pl.col("game_type") == "REG")
    )
    pos = "depth_position" if "depth_position" in d.columns else "position"
    return (
        d.filter(pl.col(pos) == "QB")
        .select([
            pl.col("season").cast(pl.Int32),
            pl.col("week").cast(pl.Int32),
            pl.col("club_code").alias("team"),
            pl.col("gsis_id").alias("opener_id"),
            pl.col("depth_team").cast(pl.Int32, strict=False).alias("depth_rank"),
        ])
        .drop_nulls(["opener_id", "depth_rank"])
    )


def _snapshot(depth_charts, schedules):
    """The dated-snapshot schema, 2025+, resolved onto team-games.

    Each club's chart is carried forward to the next game it plays: for a team-game the row kept
    is the most recent snapshot **strictly before** the game date, which is the ordering the club
    was publishing when it named the starter.
    """
    import polars as pl

    if "pos_abb" not in depth_charts.columns:
        return None
    snap = depth_charts.filter(pl.col("pos_abb") == "QB").select([
        pl.col("dt").str.slice(0, 10).str.to_date().alias("chart_date"),
        pl.col("team"),
        pl.col("gsis_id").alias("opener_id"),
        pl.col("pos_rank").cast(pl.Int32).alias("depth_rank"),
    ]).drop_nulls(["opener_id", "depth_rank", "chart_date"])
    if snap.is_empty():
        return None
    snap = canonicalize(snap, "team")

    games = pl.concat([
        schedules.filter(pl.col("game_type") == "REG").select([
            pl.col("season").cast(pl.Int32),
            pl.col("week").cast(pl.Int32),
            pl.col(f"{side}_team").alias("team"),
            pl.col("gameday").str.to_date().alias("game_date"),
        ])
        for side in ("home", "away")
    ])
    games = canonicalize(games, "team").drop_nulls("game_date")
    # only the seasons the snapshot feed actually covers
    seasons = games.filter(
        pl.col("game_date").is_between(snap["chart_date"].min(), snap["chart_date"].max())
    )["season"].unique()
    games = games.filter(pl.col("season").is_in(seasons))
    if games.is_empty():
        return None

    # Resolve the chart *date* first, then take every quarterback on that snapshot. An as-of join
    # straight onto the player rows would return a single arbitrary QB per team-game.
    dates = snap.select(["team", "chart_date"]).unique().sort(["team", "chart_date"])
    picked = (
        games.sort(["team", "game_date"])
        .join_asof(dates, left_on="game_date", right_on="chart_date", by="team",
                   strategy="backward", allow_exact_matches=False)
        .drop_nulls("chart_date")
    )
    joined = picked.join(snap, on=["team", "chart_date"], how="inner")
    return joined.select(["season", "week", "team", "opener_id", "depth_rank"])


def qb_depth(depth_charts, schedules=None):
    """``(season, week, team, qb)`` -> the club's declared depth rank for that quarterback.

    Both feed schemas are normalised to the same four columns. Team codes are canonicalised, for
    the same reason every other join in this project is: ``schedules`` and the roster feed do not
    spell St. Louis the same way.
    """
    import polars as pl

    frames = [canonicalize(_legacy(depth_charts), "team")]
    if schedules is not None:
        snap = _snapshot(depth_charts, schedules)
        if snap is not None:
            frames.append(snap)
    out = pl.concat(frames, how="vertical_relaxed")
    return out.unique(subset=["season", "week", "team", "opener_id"], keep="first")


def attach_depth(panel, depth):
    """Add ``depth_rank``, ``on_chart`` and ``chart_starter`` to a displacement panel.

    ``chart_starter`` is whether the club listed *somebody else* at rank 1 that week, which is the
    difference between "he is no longer the starter" and "the chart simply has no rank-1 row".
    """
    import polars as pl

    ranks = depth.rename({"opener_id": "_qb"})
    mine = ranks.rename({"_qb": "opener_id"})
    out = panel.join(mine, on=["season", "week", "team", "opener_id"], how="left")
    top = (
        ranks.filter(pl.col("depth_rank") == STARTER_RANK)
        .group_by(["season", "week", "team"]).agg(pl.col("_qb").first().alias("chart_qb1"))
    )
    out = out.join(top, on=["season", "week", "team"], how="left")
    # Only trust the chart for weeks it actually covers, so an unpublished week never reads as
    # "off the chart" — absence of the whole club-week is not evidence about one player.
    covered = ranks.select(["season", "week", "team"]).unique().with_columns(chart_covered=True)
    out = out.join(covered, on=["season", "week", "team"], how="left")
    return out.with_columns(
        pl.col("chart_covered").fill_null(False),
        pl.col("depth_rank").is_not_null().alias("on_chart"),
        (pl.col("chart_qb1").is_not_null() & (pl.col("chart_qb1") != pl.col("opener_id")))
        .alias("chart_replaced"),
    )
