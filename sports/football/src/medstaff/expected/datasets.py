"""The three modelling frames, and what each one's row means.

Incidence, duration and recurrence are three different questions with three different units, and
getting the unit wrong is the easiest way to produce a confident wrong answer here.

``incidence``   one row per **player-week at risk**; outcome = a spell started this week
``duration``    one row per **episode-week**; outcome = the player returned this week
``recurrence``  one row per **post-return game of exposure**; outcome = the same body part
                came back

Duration and recurrence are both person-period expansions rather than one row per episode. That
is what makes censoring correct by construction — a spell that ran out of season contributes the
weeks it actually had and then stops, instead of being dropped or counted as a clean result.
"""

from __future__ import annotations

# Covariates a club does not choose. Prior-injury history is deliberately NOT here: it is partly
# the club's own output, so it enters only via HISTORY_NUMERIC and only in the "with history"
# arm of the bound (see plan §5.3).
BASE_NUMERIC = ("age", "years_exp", "bmi", "rest_days", "week")
BASE_CATEGORICAL = ("position_group", "surface", "indoor", "short_week", "season", "is_home")

WORKLOAD_NUMERIC = ("snaps_3wk", "snap_share_3wk", "snaps_cum", "weeks_since_return")
HISTORY_NUMERIC = ("prior_designated_weeks", "prior_injury_seasons")

INCIDENCE_NUMERIC = (*BASE_NUMERIC, *WORKLOAD_NUMERIC)
INCIDENCE_CATEGORICAL = BASE_CATEGORICAL

DURATION_NUMERIC = ("age", "years_exp", "bmi", "weeks_elapsed_so_far", "games_missed_so_far",
                    "onset_week")
DURATION_CATEGORICAL = ("position_group", "body_group", "severity_worst", "is_severe",
                        "used_reserve", "season")

RECURRENCE_NUMERIC = ("age", "years_exp", "bmi", "games_missed", "weeks_elapsed",
                      "games_since_return", "return_week")
RECURRENCE_CATEGORICAL = ("position_group", "body_group", "severity_worst", "is_severe",
                          "used_reserve", "season")


def incidence_frame(exposure, episodes):
    """Player-weeks at risk, with ``onset`` marking the weeks a spell began."""
    import polars as pl

    if episodes.is_empty():
        return exposure.with_columns(pl.lit(0).alias("onset"))
    onsets = (
        episodes.select(["season", "gsis_id", pl.col("onset_week").alias("week")])
        .unique().with_columns(pl.lit(1).alias("onset"))
    )
    return (
        exposure.join(onsets, on=["season", "gsis_id", "week"], how="left")
        .with_columns(pl.col("onset").fill_null(0))
    )


def duration_frame(episodes, attributes=None):
    """One row per episode-week, outcome = returned this week.

    Censoring is handled by construction: an episode that ended for any reason other than a
    return simply contributes its weeks with ``returned`` zero throughout, rather than being
    dropped (which would bias towards fast returns) or counted as a return (which would be a lie).
    """
    import polars as pl

    if episodes.is_empty():
        return episodes

    eps = episodes.with_columns(
        (pl.col("end_week") - pl.col("onset_week") + 1).clip(lower_bound=1).alias("_n_weeks")
    )
    rows = []
    for ep in eps.iter_rows(named=True):
        n = int(ep["_n_weeks"])
        for i in range(n):
            rows.append({
                "season": ep["season"], "gsis_id": ep["gsis_id"], "team": ep["team"],
                "position_group": ep["position_group"], "body_group": ep["body_group"],
                "severity_worst": ep["severity_worst"], "is_severe": ep["is_severe"],
                "used_reserve": ep["used_reserve"], "onset_week": ep["onset_week"],
                "week": ep["onset_week"] + i,
                "weeks_elapsed_so_far": i + 1,
                "games_missed_so_far": min(i + 1, int(ep["games_missed"] or 0)),
                # only the final week of a resolved spell is a return
                "returned": int(i == n - 1 and ep["return_week"] is not None),
            })
    frame = pl.DataFrame(rows)
    if attributes is not None:
        frame = frame.join(
            attributes.select([c for c in ("season", "gsis_id", "age", "years_exp", "bmi")
                               if c in attributes.columns]),
            on=["season", "gsis_id"], how="left",
        )
    return frame


def recurrence_frame(episodes, panel, *, horizon: int, attributes=None):
    """One row per post-return game of exposure, outcome = same body part recurred.

    The risk set is **returns, not episodes** — a spell that never resolved cannot recur. Exposure
    is games the player was actually available for, so a week-17 return contributes the two games
    it had rather than being scored as a clean six-game window.
    """
    import polars as pl

    if episodes.is_empty():
        return episodes
    at_risk = episodes.filter(pl.col("at_risk"))
    if at_risk.is_empty():
        return at_risk

    played = (
        panel.filter(pl.col("team_played"))
        .select(["season", "gsis_id", "week"]).unique()
    )
    played_map: dict[tuple, list[int]] = {}
    for row in played.iter_rows(named=True):
        played_map.setdefault((row["gsis_id"], row["season"]), []).append(row["week"])
    for key in played_map:
        played_map[key].sort()

    rows = []
    for ep in at_risk.iter_rows(named=True):
        weeks = played_map.get((ep["gsis_id"], ep["season"]), [])
        exposure = [w for w in weeks if w >= ep["return_week"]][:horizon]
        recur_at = ep["recur_games_to"]
        for i, _week in enumerate(exposure):
            recurred = int(recur_at is not None and i == recur_at)
            rows.append({
                "season": ep["season"], "gsis_id": ep["gsis_id"], "team": ep["team"],
                "position_group": ep["position_group"], "body_group": ep["body_group"],
                "severity_worst": ep["severity_worst"], "is_severe": ep["is_severe"],
                "used_reserve": ep["used_reserve"], "games_missed": ep["games_missed"],
                "weeks_elapsed": ep["weeks_elapsed"], "return_week": ep["return_week"],
                "games_since_return": i + 1,
                "recurred": recurred,
            })
            if recurred:
                break  # the spell is over; later games belong to the new episode
    frame = pl.DataFrame(rows)
    if attributes is not None:
        frame = frame.join(
            attributes.select([c for c in ("season", "gsis_id", "age", "years_exp", "bmi")
                               if c in attributes.columns]),
            on=["season", "gsis_id"], how="left",
        )
    return frame


def returns_at_all_frame(episodes, attributes=None):
    """One row per episode, outcome = the player returned this season at all.

    Fitted separately from duration on purpose. Without it a club scores well on return-to-play
    simply by having its unrecovered cases quietly censored — the failure mode is gameable, so it
    gets its own outcome rather than being folded into the censor.
    """
    import polars as pl

    if episodes.is_empty():
        return episodes
    frame = episodes.select([
        "season", "gsis_id", "team", "position_group", "body_group", "severity_worst",
        "is_severe", "used_reserve", "onset_week", "games_missed", "weeks_elapsed",
        pl.col("at_risk").cast(pl.Int32).alias("returned_at_all"),
    ]).with_columns(pl.col("onset_week").alias("week"))
    if attributes is not None:
        frame = frame.join(
            attributes.select([c for c in ("season", "gsis_id", "age", "years_exp", "bmi")
                               if c in attributes.columns]),
            on=["season", "gsis_id"], how="left",
        )
    return frame


__all__ = [
    "BASE_CATEGORICAL", "BASE_NUMERIC", "DURATION_CATEGORICAL", "DURATION_NUMERIC",
    "HISTORY_NUMERIC", "INCIDENCE_CATEGORICAL", "INCIDENCE_NUMERIC", "RECURRENCE_CATEGORICAL",
    "RECURRENCE_NUMERIC", "WORKLOAD_NUMERIC",
    "duration_frame", "incidence_frame", "recurrence_frame", "returns_at_all_frame",
]
