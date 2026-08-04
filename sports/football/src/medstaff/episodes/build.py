"""Episode construction: weekly rows -> one row per continuous injury spell.

Everything downstream is computed on episodes, so this module carries the project's heaviest
correctness burden. The rules below are the ones that decide what an "injury" even is.

**Roster status is the availability signal, not snaps.** Measured on 2021-2025, players on
``RES`` (IR), ``INA``, ``DEV`` or ``CUT`` take a snap in ~0.02% of player-weeks, so weekly
roster status separates played from did-not-play almost perfectly — while being gsis-keyed with
no crosswalk loss and reaching back to 2002. ``snap_counts`` keys on ``pfr_player_id`` and
crosswalks at only 81.7% from 2012, and it would misread the 21% of active player-weeks with
zero snaps: a healthy scratch or a deep backup is **available**, not injured. Snaps are left to
the exposure stage as an intensity covariate.

**What terminates an episode.** Only an ``ACT`` week with no injury designation — i.e. the
player is back on the active roster and off the report. Weeks that are ``INA``/``RES``, on the
report, or a bye all *continue* an open spell. That handles report noise for free: a player
listed in weeks 5 and 7 but merely inactive in week 6 never "returned" in week 6.

**What starts one.** Only an *impaired* week — a real body part on the report, or reserve
status. A plain ``INA`` week does not start an episode, because a healthy scratch is not an
injury; it only extends a spell already open.

**Concurrent injuries are not separable in this source, so they are not split.** The report
carries one *primary* body part per player-week, and clubs flip it week to week as the picture
changes. Splitting on that flip would manufacture two episodes out of one spell far more often
than it would catch a genuine second injury. A spell therefore keeps the body group it opened
with, records every group seen in ``groups_seen``, and sets ``group_changed`` — which downstream
can filter on. The cost is that a true simultaneous second injury is absorbed into the first;
the alternative cost was inventing episodes, which is worse for every rate this project reports.

**Body part on reserve weeks carries forward *within* a spell, never across one.** A player on
IR usually drops off the report entirely, so an open spell keeps the group it already has and a
spell that opens on a bare reserve week adopts the first group it later sees, falling back to
``unknown``. A global forward-fill was rejected: it would let an unrelated IR stint in week 10
inherit the knee from week 2, across a return, and long IR spells are exactly the ones that
matter most.
"""

from __future__ import annotations


# Week states, in the priority order they are resolved.
AVAILABLE = "available"        # ACT and undesignated -> terminates an open episode
IMPAIRED = "impaired"          # designated with a real body part, or on reserve
OUT_OTHER = "out_other"        # inactive/other absence with no injury designation
PRACTICE_SQUAD = "practice_squad"
BYE = "bye"
OFF_ROSTER = "off_roster"

# Practice participation is the severity proxy. It is regime-invariant (0-1% null in every
# season), unlike report_status, which is ~half null from 2016 — see MEDSTAFF_PLAN §2.2.
SEVERITY = {"did not participate": "DNP", "limited": "LIMITED", "full": "FULL"}
SEVERITY_ORDER = ("FULL", "LIMITED", "DNP")

# Taxonomy groups that are not body parts. A spell must never be labelled with one.
NON_BODY_GROUPS = ("non_injury", "illness")

CENSOR_SEASON_END = "season_end"
CENSOR_TEAM_CHANGE = "team_change"
CENSOR_OFF_ROSTER = "off_roster"
CENSOR_PRACTICE_SQUAD = "practice_squad"


def practice_severity(status: str | None) -> str | None:
    """Map a practice-participation string to FULL / LIMITED / DNP."""
    if status is None:
        return None
    text = str(status).strip().lower()
    if not text:
        return None
    for needle, code in SEVERITY.items():
        if needle in text:
            return code
    return None


def team_week_games(schedules):
    """(season, week, team) for every regular-season game — the bye-week reference.

    A week absent from this table is a bye for that club: the spell continues but no *game* was
    missed, which is why byes are excluded from the games-missed count but not from elapsed time.
    """
    import polars as pl

    reg = schedules.filter(pl.col("game_type") == "REG")
    sides = [
        reg.select(["season", "week", pl.col(side).alias("team")])
        for side in ("home_team", "away_team")
    ]
    return pl.concat(sides).unique().with_columns(pl.lit(True).alias("team_played"))


def build_panel(rosters_weekly, injuries, schedules):
    """One row per (season, week, gsis_id) with the week's state resolved.

    The roster is the universe — it is the only source that covers every position for every week
    — with the injury report joined on for body part and severity.
    """
    import polars as pl

    ros = rosters_weekly.filter(pl.col("game_type") == "REG")
    inj = injuries.filter(pl.col("game_type") == "REG")

    # Defensive dedupe: one report row per player-week, keeping the latest revision. Measured at
    # zero duplicates for 2023, but a duplicated row would silently double an episode's evidence.
    if "date_modified" in inj.columns:
        inj = inj.sort("date_modified").unique(
            subset=["season", "week", "gsis_id"], keep="last", maintain_order=True
        )
    else:
        inj = inj.unique(subset=["season", "week", "gsis_id"], keep="last")

    inj_cols = inj.select([
        "season", "week", "gsis_id",
        pl.col("body_part_group").alias("report_body_group"),
        pl.col("body_part_raw").alias("report_body_raw"),
        "is_severe", "practice_status", "report_status",
    ])

    panel = (
        ros.join(inj_cols, on=["season", "week", "gsis_id"], how="left")
        .join(team_week_games(schedules), on=["season", "week", "team"], how="left")
        .with_columns(pl.col("team_played").fill_null(False))
    )

    # A designated week needs a *real* body part: "non_injury" and nulls do not count. This is
    # where resting players, illness-free personal absences and suspensions drop out.
    designated = (
        pl.col("report_body_group").is_not_null()
        & ~pl.col("report_body_group").is_in(["non_injury", "illness"])
    )
    panel = panel.with_columns(
        designated.alias("designated"),
        pl.col("practice_status").map_elements(practice_severity, return_dtype=pl.String)
        .alias("severity"),
    )

    state = (
        pl.when(pl.col("on_practice_squad")).then(pl.lit(PRACTICE_SQUAD))
        .when(pl.col("on_reserve") | pl.col("designated")).then(pl.lit(IMPAIRED))
        .when(~pl.col("team_played")).then(pl.lit(BYE))
        .when(pl.col("active")).then(pl.lit(AVAILABLE))
        .otherwise(pl.lit(OUT_OTHER))
    )
    return panel.with_columns(state.alias("week_state")).sort(["gsis_id", "season", "week"])


def _episode_rows(rows):
    """Walk one player-season's weeks in order, emitting episodes.

    Written as an explicit pass rather than a vectorised group-by: the state machine has enough
    exits (return, season end, trade, practice-squad move, release) that each is clearer named
    than encoded, and each has a test.
    """
    out = []
    current = None

    def close(reason, end_week, return_week=None):
        nonlocal current
        if current is None:
            return
        current["end_week"] = end_week
        current["return_week"] = return_week
        current["censored"] = return_week is None
        current["censor_reason"] = reason
        current["never_returned"] = return_week is None
        current["weeks_elapsed"] = end_week - current["onset_week"] + 1
        out.append(current)
        current = None

    prev_team = None
    prev_week = None
    for row in rows:
        week, state, team = row["week"], row["week_state"], row["team"]

        if current is not None:
            if prev_week is not None and week - prev_week > 1:
                # Weeks absent from the roster entirely — released, then re-signed. The club
                # was not rehabbing him in between, so the spell cannot be credited across it.
                close(CENSOR_OFF_ROSTER, prev_week)
            elif team != prev_team:
                # Rehab credit belongs to whoever did the rehab: the onset club's spell ends and
                # a fresh one opens below for the acquiring club if he is still impaired.
                close(CENSOR_TEAM_CHANGE, prev_week)
        prev_team, prev_week = team, week

        if state == AVAILABLE:
            close(None, week - 1, return_week=week)
            continue
        if state == PRACTICE_SQUAD:
            close(CENSOR_PRACTICE_SQUAD, week - 1)
            continue

        if state == IMPAIRED:
            group = row["report_body_group"]
            if current is None:
                current = {
                    "season": row["season"], "gsis_id": row["gsis_id"], "team": team,
                    "position": row.get("position"),
                    "position_group": row.get("position_group"), "side": row.get("side"),
                    "body_group": (group if group not in NON_BODY_GROUPS else None),
                    "body_raw_onset": row.get("report_body_raw"),
                    "onset_week": week, "severity_onset": row.get("severity"),
                    "severity_worst": row.get("severity"),
                    "is_severe": bool(row.get("is_severe") or False),
                    "games_missed": 0, "n_byes": 0, "weeks_on_reserve": 0,
                    "used_reserve": False, "groups_seen": set(),
                }
            # A reserve week makes the week impaired regardless of what the report said, so a
            # spell can pick up an "illness"/"non_injury" label from a stray row. Those are not
            # body parts; leave the spell unlabelled rather than mislabel it.
            if group is not None and group not in NON_BODY_GROUPS:
                if current["body_group"] is None:
                    current["body_group"] = group
                current["groups_seen"].add(group)
            # Severity and the severe-word flag are the worst seen across the spell, not just
            # at onset: clubs frequently list a knock before the diagnosis firms up.
            sev = row.get("severity")
            if sev is not None:
                worst = current["severity_worst"]
                if worst is None or SEVERITY_ORDER.index(sev) > SEVERITY_ORDER.index(worst):
                    current["severity_worst"] = sev
            current["is_severe"] = current["is_severe"] or bool(row.get("is_severe") or False)
            if row["on_reserve"]:
                current["used_reserve"] = True
                current["weeks_on_reserve"] += 1
            if row["team_played"]:
                current["games_missed"] += 1
            continue

        # BYE / OUT_OTHER: extends an open spell, never starts one. A healthy scratch is not an
        # injury, but an inactive week inside an open spell is still a game missed.
        if current is not None:
            if state == BYE:
                current["n_byes"] += 1
            elif row["team_played"]:
                current["games_missed"] += 1

    if current is not None:
        close(CENSOR_SEASON_END, rows[-1]["week"])
    return out


def build_episodes(panel):
    """One row per injury spell, from a panel produced by :func:`build_panel`."""
    import polars as pl

    keep = [
        "season", "week", "gsis_id", "team", "position", "position_group", "side",
        "week_state", "report_body_group", "report_body_raw", "is_severe", "severity",
        "on_reserve", "team_played",
    ]
    frame = panel.select([c for c in keep if c in panel.columns]).sort(
        ["gsis_id", "season", "week"]
    )

    episodes = []
    for (_gsis, _season), group in frame.group_by(["gsis_id", "season"], maintain_order=True):
        episodes.extend(_episode_rows(group.to_dicts()))

    if not episodes:
        return pl.DataFrame()

    for ep in episodes:
        seen = ep.pop("groups_seen")
        ep["group_changed"] = len(seen) > 1
        ep["body_group"] = ep["body_group"] or "unknown"
    return pl.DataFrame(episodes).sort(["season", "gsis_id", "onset_week"])


__all__ = [
    "AVAILABLE", "BYE", "CENSOR_OFF_ROSTER", "CENSOR_PRACTICE_SQUAD", "CENSOR_SEASON_END",
    "CENSOR_TEAM_CHANGE", "IMPAIRED", "OFF_ROSTER", "OUT_OTHER", "PRACTICE_SQUAD",
    "NON_BODY_GROUPS", "SEVERITY_ORDER", "build_episodes", "build_panel", "practice_severity",
    "team_week_games",
]
