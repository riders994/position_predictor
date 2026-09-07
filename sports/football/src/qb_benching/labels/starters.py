"""Who started each team-game, and who opened the season.

⚠️ ``schedules`` and ``rosters_weekly`` do not spell teams the same way. The 2015 Rams are
``STL`` in the schedule and ``LA`` in the roster table, and an uncanonicalised comparison of the
two reads every Rams quarterback as playing for another club — which classified Nick Foles's
2015 benching as a departure. Both sides of every team join are canonicalised through
:func:`medstaff.data.teams.canonicalize`, which already carries the relocation and spelling map;
this project deliberately does not add a third copy of it.

The source of truth is ``schedules``, which carries ``home_qb_id`` / ``away_qb_id`` — nflverse's
record of the **announced starter**, populated for 100% of regular-season games 1999-2025. That
is the right column for this project: the question is who the club chose to start, which is not
the same as who ended up throwing the most passes. A starter pulled at halftime is still the
starter that week, and counting him as displaced would fold in-game hooks into a label meant to
capture a week-to-week decision.

``starter_disagreements`` measures the gap against the obvious alternative — the QB with the most
dropbacks in the game — and it is small: **96.7% agreement over 13,183 team-games**. The residual
is the interesting 3%, and it is a diagnostic in the stage-1 report rather than something to
smooth over.
"""

from __future__ import annotations

from medstaff.data.teams import canonicalize


def game_starters(schedules):
    """One row per ``(season, week, team)``: the announced starting QB.

    Unpivots the home/away pair into a team-per-row table. Only regular-season games appear, so a
    week absent for a club is a bye and simply never enters the panel — the reason this project
    never needs to reason about bye weeks explicitly, unlike ``medstaff``, whose panel is keyed on
    the roster rather than on games.
    """
    import polars as pl

    reg = schedules.filter(pl.col("game_type") == "REG")
    sides = [
        reg.select([
            pl.col("season").cast(pl.Int32),
            pl.col("week").cast(pl.Int32),
            pl.col(f"{side}_team").alias("team"),
            pl.col(f"{side}_qb_id").alias("starter_id"),
            pl.col(f"{side}_qb_name").alias("starter_name"),
        ])
        for side in ("home", "away")
    ]
    return canonicalize(pl.concat(sides), "team").sort(["season", "team", "week"])


def opening_starters(starters):
    """One row per ``(season, team)``: the QB who started the club's **first game**.

    Keyed on the first game actually played rather than ``week == 1``, so a season whose opening
    week is disturbed still resolves. 2001 is the case that matters — the week-2 slate was
    postponed after September 11 and replayed at the end of the season — and a literal
    ``week == 1`` filter would be right there but wrong in principle.
    """
    import polars as pl

    known = starters.filter(pl.col("starter_id").is_not_null())
    return (
        known.sort("week")
        .group_by(["season", "team"])
        .first()
        .select([
            "season", "team",
            pl.col("starter_id").alias("opener_id"),
            pl.col("starter_name").alias("opener_name"),
            pl.col("week").alias("opening_week"),
        ])
        .sort(["season", "team"])
    )


def dropback_starters(weekly, *, min_dropbacks: int = 5):
    """The alternative definition: the QB with the most dropbacks in a team-game.

    Used only to measure how far ``schedules`` can be trusted. ``min_dropbacks`` drops team-games
    where no QB threw meaningfully — a wildcat game, or a row that is an artefact.
    """
    import polars as pl

    w = weekly.filter(
        (pl.col("season_type") == "REG") & (pl.col("position") == "QB")
    ).select([
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int32),
        pl.col("recent_team").alias("team"),
        pl.col("player_id").alias("dropback_id"),
        (pl.col("attempts").fill_null(0) + pl.col("sacks").fill_null(0)).alias("dropbacks"),
    ])
    return (
        canonicalize(w, "team")
        .sort("dropbacks", descending=True)
        .group_by(["season", "week", "team"])
        .first()
        .filter(pl.col("dropbacks") >= min_dropbacks)
    )


def starter_disagreements(starters, weekly, *, min_dropbacks: int = 5):
    """Team-games where the announced starter is not the QB who took the most dropbacks.

    Returns the joined frame with an ``agrees`` flag, so the caller can both quote the agreement
    rate and list the disagreements. These are overwhelmingly in-game changes — a hook or an
    injury — which is exactly the class of event the label is designed *not* to count.
    """
    import polars as pl

    db = dropback_starters(weekly, min_dropbacks=min_dropbacks)
    joined = starters.join(db, on=["season", "week", "team"], how="inner").filter(
        pl.col("starter_id").is_not_null()
    )
    return joined.with_columns((pl.col("starter_id") == pl.col("dropback_id")).alias("agrees"))
