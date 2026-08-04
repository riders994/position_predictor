"""Attaching staff identity to a club-window grade, and saying plainly when it cannot be done.

nflverse publishes head coaches, not head athletic trainers, and no free structured source for
trainer-by-team-season exists. So the grade is a property of the **franchise over the window**
unless a table is supplied. :func:`attach_staff` is the hook for that table; without it the
report says exactly what it is grading rather than implying more.

Head-coach change is the only turnover instrument available, and it is used to **annotate**
windows, never to split them: a window short enough to sit inside one coaching tenure is too
short to grade.
"""

from __future__ import annotations


def head_coach_by_season(schedules):
    """Per (season, team) head coach — the club's most-frequent coach that season."""
    import polars as pl

    reg = schedules.filter(pl.col("game_type") == "REG")
    sides = [
        reg.select(["season", pl.col(f"{s}_team").alias("team"),
                    pl.col(f"{s}_coach").alias("coach")])
        for s in ("home", "away")
    ]
    return (
        pl.concat(sides).drop_nulls("coach")
        .group_by(["season", "team", "coach"]).len()
        .sort("len", descending=True)
        .unique(subset=["season", "team"], keep="first")
        .drop("len").sort(["team", "season"])
    )


def coach_changes(coaches):
    """Per team: how many distinct head coaches, and whether the window spans a change."""
    import polars as pl

    return (
        coaches.group_by("team").agg(
            pl.col("coach").n_unique().alias("n_coaches"),
            pl.col("coach").sort_by("season").first().alias("coach_first"),
            pl.col("coach").sort_by("season").last().alias("coach_last"),
        )
        .with_columns((pl.col("n_coaches") > 1).alias("regime_changed"))
        .sort("team")
    )


def attach_staff(grades, staff_table=None, *, team_col="team"):
    """Join a supplied ``(team, head_athletic_trainer, ...)`` table if one is given.

    A no-op without a table, by design — the alternative is inventing an attribution the data
    does not support.
    """
    if staff_table is None:
        return grades
    import polars as pl

    from ..data.teams import canonicalize

    table = canonicalize(staff_table, team_col)
    if hasattr(grades, "join"):
        keys = [c for c in (team_col, "season") if c in table.columns and c in grades.columns]
        return grades.join(table, on=keys or [team_col], how="left")
    return grades.merge(table.to_pandas() if isinstance(table, pl.DataFrame) else table,
                        on=team_col, how="left")


def load_staff_table(path):
    """Read a staff CSV, canonicalising team codes on the way in."""
    import polars as pl

    from ..data.teams import canonicalize

    return canonicalize(pl.read_csv(path), "team")


__all__ = ["attach_staff", "coach_changes", "head_coach_by_season", "load_staff_table"]
