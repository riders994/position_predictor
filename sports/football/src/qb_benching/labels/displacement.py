"""Every week the season's opening starter did not start, and why.

Three reasons, and the whole point of the project is that they are different questions:

``injured``
    Hurt: listed on that week's injury report, or on a reserve list.
``gone``
    No longer the club's to start: released, traded, on the practice squad, or suspended.
``benched``
    Everything else. He was with the club, nothing said he was hurt, and somebody else started.
    This is the class the project models, and it is the **residual** of the ladder below — every
    week the data can explain another way is explained another way, so the class is a lower bound.

⚠️ The roster table encodes injured reserve two incompatible ways
----------------------------------------------------------------
``medstaff`` sets ``FIRST_COMPARABLE_SEASON = 2021`` because ``rosters_weekly``'s ACT/INA split is
unpopulated before then. For quarterbacks the problem is worse and differently shaped, and it was
found here by a label that reported **zero benchings in 2015** — a season in which Colin
Kaepernick visibly lost the San Francisco job to Blaine Gabbert.

The cause: **before 2021 ``status`` is not a weekly value at all.** It is the player's
season-final status stamped on every one of his weeks. Only 1-4% of pre-2021 QB-seasons have a
status that varies by week, against 50-72% from 2021 on. Kaepernick started the first eight games
of 2015 and finished the year on injured reserve, so all ten of his rows read ``RES`` — including
the weeks he was starting, and including the weeks he was benched and healthy.

What *is* informative before 2021 is **whether a row exists at all**. When a player goes on
reserve his rows stop: **74.6%** of pre-2021 QB-seasons ending in ``RES`` are truncated before the
end of the season, against **0%** from 2021 on, where reserve players keep weekly ``RES`` rows all
year. Kaepernick's rows stop at week 10, which is when he went on IR — so presence recovers what
the status value lost. Availability is therefore read from **presence before 2021 and from the
status value from 2021 on**, and the ladder below is explicit about which evidence resolved each
week so the report can show the mix.

**Known residual.** The quarter of pre-2021 reserve spells that do *not* truncate cannot be seen
this way and will read as benched. That inflates the pre-2021 benched class, and
``evidence_mix`` reports the exposure rather than hiding it.

⚠️ Injured reserve is invisible to the injury report
---------------------------------------------------
A player placed on IR drops off the weekly report entirely, so the report alone cannot carry the
label. Measured against the trustworthy 2021+ status, the weeks with no report listing and no
appearance split **195 reserve / 104 active / 20 inactive** — roughly 60% genuinely hurt. A label
keyed on the report alone would call all of them benchings.

**The label is a lower bound, by choice.** Any appearance on the report counts as injured,
including a Questionable listing a club may have used to dress up a benching. Counting only Out
and Doubtful would read more benchings but depends on ``report_status``, which is roughly half
null from 2016 when the league dropped "Probable". ``on_report_status`` is carried through so the
looser rule can be run as a sensitivity.
"""

from __future__ import annotations

from medstaff.data.teams import canonicalize

#: Reserve lists — unavailable for a physical reason, not a competitive one.
RESERVE_STATUSES = ("RES", "PUP", "NFI", "RSN", "RSR")

#: Not the club's to start. ``DEV`` is the practice squad, ``SUS`` a suspension, ``TRC``/``TRT``
#: trade codes. None is a benching (nobody preferred the backup on the merits) and none is an
#: injury, so they are their own class rather than being forced into one of the other two.
AWAY_STATUSES = ("CUT", "TRD", "TRT", "TRC", "UFA", "DEV", "SUS")

#: From this season ``rosters_weekly.status`` is a genuine weekly value. Before it, the column is
#: a season-final stamp and only row *presence* is informative. See the module docstring.
ROSTER_REGIME_SEASON = 2021

HELD = "held"
BENCHED = "benched"
INJURED = "injured"
GONE = "gone"


def _roster_presence(rosters_weekly):
    """``(season, week, qb)`` -> was he on the roster table that week, with what status and club."""
    import polars as pl

    r = rosters_weekly
    if "game_type" in r.columns:
        r = r.filter(pl.col("game_type") == "REG")
    return (
        r.filter(pl.col("position") == "QB")
        .select([
            pl.col("season").cast(pl.Int32),
            pl.col("week").cast(pl.Int32),
            pl.col("gsis_id").alias("opener_id"),
            pl.col("status").alias("roster_status"),
            pl.col("team").alias("roster_team"),
        ])
        .pipe(canonicalize, "roster_team")
        .unique(subset=["season", "week", "opener_id"], keep="first")
        .with_columns(on_roster=True)
    )


def _final_status(rosters_weekly):
    """``(season, qb)`` -> the status on his last roster week.

    Before 2021 this is the value stamped on every week, so it is only meaningful as what it
    actually is: how the season ended. It resolves *why* a player who has left the roster is
    absent — hurt, or gone — once presence has established that he left.
    """
    import polars as pl

    r = rosters_weekly
    if "game_type" in r.columns:
        r = r.filter(pl.col("game_type") == "REG")
    return (
        r.filter(pl.col("position") == "QB")
        .select([
            pl.col("season").cast(pl.Int32),
            pl.col("week").cast(pl.Int32),
            pl.col("gsis_id").alias("opener_id"),
            pl.col("status"),
        ])
        .sort("week")
        .group_by(["season", "opener_id"])
        .last()
        .select(["season", "opener_id", pl.col("status").alias("final_status")])
    )


def _appearances(weekly):
    """``(season, week, qb)`` -> he took a snap that produced a stat line."""
    import polars as pl

    return (
        weekly.filter(pl.col("season_type") == "REG")
        .select([
            pl.col("season").cast(pl.Int32),
            pl.col("week").cast(pl.Int32),
            pl.col("player_id").alias("opener_id"),
        ])
        .unique()
        .with_columns(appeared=True)
    )


def _weekly_report(injuries):
    """``(season, week, qb)`` -> was he on the injury report, and with what designation."""
    import polars as pl

    inj = injuries
    if "game_type" in inj.columns:
        inj = inj.filter(pl.col("game_type") == "REG")
    return (
        inj.filter(pl.col("position") == "QB")
        .select([
            pl.col("season").cast(pl.Int32),
            pl.col("week").cast(pl.Int32),
            pl.col("gsis_id").alias("opener_id"),
            pl.col("report_status").alias("on_report_status"),
        ])
        .unique(subset=["season", "week", "opener_id"], keep="first")
        .with_columns(on_report=True)
    )


def displacement_panel(starters, openers, rosters_weekly, injuries, weekly=None):
    """One row per team-game **after** the opener, with the opener's fate that week.

    The grid is the club's actual games, taken from ``starters``, so byes and the move from 16 to
    17 games never fabricate a week. Games at or before the club's opening week are excluded — he
    started that one by construction.

    ``weekly`` is optional and adds ``appeared`` / ``started_again``, which classify nothing but
    let the report say how many residual benchings are corroborated by the quarterback playing
    again afterwards.
    """
    import polars as pl

    panel = starters.join(openers, on=["season", "team"], how="inner")
    panel = panel.filter(pl.col("week") > pl.col("opening_week"))
    panel = panel.join(_roster_presence(rosters_weekly), on=["season", "week", "opener_id"],
                       how="left")
    panel = panel.join(_final_status(rosters_weekly), on=["season", "opener_id"], how="left")
    panel = panel.join(_weekly_report(injuries), on=["season", "week", "opener_id"], how="left")
    panel = panel.with_columns(
        pl.col("on_report").fill_null(False),
        pl.col("on_roster").fill_null(False),
        (pl.col("starter_id") == pl.col("opener_id")).alias("held"),
    )
    if weekly is not None:
        panel = panel.join(_appearances(weekly), on=["season", "week", "opener_id"], how="left")
        panel = panel.with_columns(pl.col("appeared").fill_null(False))
    # did he start for this club again later in the season? corroborates a benching
    later = (
        panel.filter(pl.col("held"))
        .group_by(["season", "team"]).agg(pl.col("week").max().alias("_last_held_week"))
    )
    panel = panel.join(later, on=["season", "team"], how="left")
    panel = panel.with_columns(
        (pl.col("_last_held_week").is_not_null() & (pl.col("week") < pl.col("_last_held_week")))
        .alias("started_again")
    ).drop("_last_held_week")
    return panel.sort(["season", "team", "week"])


def classify_displacement(panel, *, regime_season: int = ROSTER_REGIME_SEASON):
    """Add ``outcome`` and ``evidence`` — which rule resolved each week.

    Where the club published a depth chart, the chart decides, because it records the two facts
    the label needs directly rather than by elimination: a quarterback listed behind someone else
    has been **demoted**, and one absent from his club's chart is **unavailable**. The roster
    ladder below it is the fallback for weeks the chart does not cover.

    The ladder, in order:

    1. he started — ``held``
    2. listed on the injury report — ``injured`` (``evidence="injury_report"``). Ahead of the
       chart deliberately: a quarterback who is both listed hurt and demoted is counted hurt,
       which keeps benching a lower bound.
    3. absent from a chart that covers his club that week — unavailable, and the roster says
       whether that is reserve or a departure (``evidence="off_chart"``)
    4. on the chart with another quarterback at rank 1 — ``benched`` (``evidence="depth_chart"``)
    5. not on the roster table that week — his season-final status decides
       (``evidence="off_roster"``)
    6. rostered by another club — ``gone`` (``evidence="other_club"``)
    7. from 2021, a reserve or departure status on the week's own row
       (``evidence="weekly_status"``)
    8. otherwise — ``benched`` (``evidence="residual"``)

    Step 7 is restricted to the era where ``status`` is a weekly value. Applying it earlier is
    what produced a zero-benching 2015: Kaepernick's benched weeks carry the ``RES`` he ended the
    season on.
    """
    import polars as pl

    modern = pl.col("season") >= regime_season
    reserve_now = pl.col("roster_status").is_in(RESERVE_STATUSES)
    away_now = pl.col("roster_status").is_in(AWAY_STATUSES)
    left_club = pl.col("roster_team").is_not_null() & (pl.col("roster_team") != pl.col("team"))
    final_reserve = pl.col("final_status").is_in(RESERVE_STATUSES)
    final_away = pl.col("final_status").is_in(AWAY_STATUSES)

    has_chart = "chart_covered" in panel.columns
    if has_chart:
        covered = pl.col("chart_covered")
        off_chart = covered & ~pl.col("on_chart")
        demoted = covered & pl.col("on_chart") & pl.col("chart_replaced")
        # an absent quarterback who is also off the roster, or whose season ended on a departure,
        # left the club; otherwise absence from the chart is a reserve-list move
        off_chart_gone = off_chart & (left_club | away_now | (~pl.col("on_roster") & final_away))
    else:
        off_chart = demoted = off_chart_gone = pl.lit(False)

    outcome = (
        pl.when(pl.col("held")).then(pl.lit(HELD))
        .when(pl.col("on_report")).then(pl.lit(INJURED))
        .when(off_chart_gone).then(pl.lit(GONE))
        .when(off_chart).then(pl.lit(INJURED))
        .when(demoted).then(pl.lit(BENCHED))
        .when(~pl.col("on_roster") & final_reserve).then(pl.lit(INJURED))
        .when(~pl.col("on_roster")).then(pl.lit(GONE))
        .when(left_club).then(pl.lit(GONE))
        .when(modern & reserve_now).then(pl.lit(INJURED))
        .when(modern & away_now).then(pl.lit(GONE))
        .otherwise(pl.lit(BENCHED))
    )
    evidence = (
        pl.when(pl.col("held")).then(pl.lit("held"))
        .when(pl.col("on_report")).then(pl.lit("injury_report"))
        .when(off_chart).then(pl.lit("off_chart"))
        .when(demoted).then(pl.lit("depth_chart"))
        .when(~pl.col("on_roster")).then(pl.lit("off_roster"))
        .when(left_club).then(pl.lit("other_club"))
        .when(modern & (reserve_now | away_now)).then(pl.lit("weekly_status"))
        .otherwise(pl.lit("residual"))
    )
    return panel.with_columns(outcome.alias("outcome"), evidence.alias("evidence"))


def evidence_mix(classified):
    """How each displaced week was resolved, by era — the exposure to the roster-table break."""
    import polars as pl

    return (
        classified.filter(~pl.col("held"))
        .with_columns(
            pl.when(pl.col("season") >= ROSTER_REGIME_SEASON)
            .then(pl.lit(f"{ROSTER_REGIME_SEASON}+"))
            .otherwise(pl.lit(f"pre-{ROSTER_REGIME_SEASON}")).alias("era")
        )
        .group_by(["era", "evidence", "outcome"]).agg(pl.len().alias("team_games"))
        .sort(["era", "team_games"], descending=[False, True])
    )
