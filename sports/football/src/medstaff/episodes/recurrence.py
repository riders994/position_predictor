"""Recurrence: did the same body part come back after the player returned?

This is the project's most attributable outcome, so its definition does more work than any
other. Three choices carry it:

**The clock counts games the player was available for, not calendar weeks.** A player who
returns in week 17 has two games of exposure, not "six weeks" of it; counting calendar time
would score a late-season return as a clean result purely because the season ran out. Counting
available games makes a short window *less exposure* rather than a free pass, and it pauses
across byes and the offseason automatically.

**Attribution is to the club of the return** — the staff that cleared him to play. Not the club
of onset: if a player is traded while hurt, the decision that matters for recurrence is made by
whoever put him back on the field.

**The risk set is returns, not episodes.** An episode that never resolved cannot recur, and
including it would count unresolved injuries as clean outcomes.
"""

from __future__ import annotations

# Games-of-exposure windows. K=6 is primary; the others are reported as sensitivity because
# there is no principled threshold and the answer should not hinge on one.
HORIZONS = {"k3": 3, "k6": 6, "k12": 12}
PRIMARY_HORIZON = "k6"


def available_games_after(panel, *, season, gsis_id, after_week):
    """Weeks after ``after_week`` in which the club played and the player was on the roster.

    "Available" here means rostered for a game that was played — the exposure denominator a
    recurrence could have happened in.
    """
    import polars as pl

    rows = panel.filter(
        (pl.col("season") == season)
        & (pl.col("gsis_id") == gsis_id)
        & (pl.col("week") > after_week)
        & pl.col("team_played")
    ).sort("week")
    return rows["week"].to_list()


def attach_recurrence(episodes, panel, *, horizons=None):
    """Flag, for each returned episode, whether the same body group recurred.

    Adds one boolean per horizon plus ``recur_games_to`` (games of exposure until the recurrence,
    null if none), ``recur_same_season``, ``recur_next_season`` and ``return_team``.
    """
    import polars as pl

    horizons = horizons or HORIZONS
    if episodes.is_empty():
        return episodes

    eps = episodes.sort(["gsis_id", "season", "onset_week"])
    # Exposure lookup: the weeks each player's club actually played, per season.
    played = (
        panel.filter(pl.col("team_played"))
        .select(["season", "gsis_id", "week"]).unique()
        .sort(["gsis_id", "season", "week"])
    )
    played_map: dict[tuple, list[int]] = {}
    for row in played.iter_rows(named=True):
        played_map.setdefault((row["gsis_id"], row["season"]), []).append(row["week"])

    by_player: dict[str, list[dict]] = {}
    for ep in eps.iter_rows(named=True):
        by_player.setdefault(ep["gsis_id"], []).append(ep)

    out = []
    for gsis, spells in by_player.items():
        for i, ep in enumerate(spells):
            rec = {k: None for k in horizons}
            games_to = None
            same_season = False
            next_season = False

            if ep["return_week"] is not None:
                weeks = played_map.get((gsis, ep["season"]), [])
                exposure = [w for w in weeks if w >= ep["return_week"]]
                for later in spells[i + 1:]:
                    if later["body_group"] != ep["body_group"]:
                        continue
                    if later["season"] == ep["season"]:
                        if later["onset_week"] <= ep["return_week"]:
                            continue
                        # games of exposure between the return and the new onset
                        games_to = sum(1 for w in exposure if w < later["onset_week"])
                        same_season = True
                        break
                    if later["season"] == ep["season"] + 1:
                        next_season = True
                        break
                for name, k in horizons.items():
                    rec[name] = bool(same_season and games_to is not None and games_to <= k)

            row = dict(ep)
            row.update({
                f"recur_{name}": (None if ep["return_week"] is None else bool(v))
                for name, v in rec.items()
            })
            row["recur_games_to"] = games_to
            row["recur_same_season"] = None if ep["return_week"] is None else same_season
            row["recur_next_season"] = None if ep["return_week"] is None else next_season
            row["at_risk"] = ep["return_week"] is not None
            out.append(row)

    return pl.DataFrame(out).sort(["season", "gsis_id", "onset_week"])


def recurrence_rates(episodes, *, by=None, horizon=PRIMARY_HORIZON):
    """Recurrence rate over the at-risk (returned) episodes only."""
    import polars as pl

    col = f"recur_{horizon}"
    at_risk = episodes.filter(pl.col("at_risk"))
    group = by or []
    if not group:
        return at_risk.select(
            pl.len().alias("returns"),
            pl.col(col).sum().alias("recurrences"),
            pl.col(col).mean().alias("rate"),
        )
    return (
        at_risk.group_by(group)
        .agg(
            pl.len().alias("returns"),
            pl.col(col).sum().alias("recurrences"),
            pl.col(col).mean().alias("rate"),
        )
        .sort(group)
    )


__all__ = [
    "HORIZONS", "PRIMARY_HORIZON", "attach_recurrence", "available_games_after",
    "recurrence_rates",
]
