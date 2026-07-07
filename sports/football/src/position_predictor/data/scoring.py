"""League-specific fantasy-point formulas, computed per player-game from raw weekly box scores.

nflverse ships ``fantasy_points_ppr`` (standard PPR: 4-pt passing TD, no yardage bonuses,
-2/fumble lost) as the default target. A real league's scoring can diverge from that —
this module recomputes an exact point total from box-score columns for leagues that do.

Add a new league by adding a formula function here and wiring its name into
:func:`compute_points`.
"""
from __future__ import annotations

RTSPORTS_INPUT_COLS = [
    "passing_yards", "passing_tds", "passing_2pt_conversions", "interceptions",
    "rushing_yards", "rushing_tds", "rushing_2pt_conversions",
    "receptions", "receiving_yards", "receiving_tds", "receiving_2pt_conversions",
    "special_teams_tds",
    "fg_made_0_19", "fg_made_20_29", "fg_made_30_39", "fg_made_40_49", "fg_made_50_59",
    "fg_made_60_", "pat_made",
]


def compute_rtsports_points(weekly):
    """Per-row (player-game) RT Sports league point total.

    Confirmed against the league by-laws (2026-07-06): 1-pt PPR, 6-pt passing/rushing/
    receiving TDs (passing TD is 6, not the nflverse-standard 4), -2/INT thrown, +2/2-pt
    conversion (any type), +5 bonus for 100+ rushing yds, +5 bonus for 100+ receiving yds,
    +5 bonus for 300+ passing yds, +6 for a kickoff/punt return TD (``special_teams_tds``).
    Fumbles are **not** penalized for the offensive player in this league (recoveries only
    score the DST side), so — unlike nflverse's standard formula — no fumble term here.

    Kicker (K): distance-tiered FG scoring + PAT, additive with everything above (a K row's
    passing/rushing/receiving columns are all zero, so this doesn't need a position branch).
    nflverse's ``fg_made_60_`` bucket is its most granular (60+ yds) — the by-laws' 70-89 yd
    tiers are unreachable in practice (no NFL kicker has ever attempted a 70+ yd FG), so 60+
    is scored flat at the 60-69 yd rate. No penalty for a miss (by-laws didn't list one).
    """
    def col(name):
        return weekly[name] if name in weekly.columns else 0

    passing = (col("passing_yards") * 0.04
               + col("passing_tds") * 6
               + col("passing_2pt_conversions") * 2
               - col("interceptions") * 2
               + (col("passing_yards") >= 300) * 5)
    rushing = (col("rushing_yards") * 0.10
               + col("rushing_tds") * 6
               + col("rushing_2pt_conversions") * 2
               + (col("rushing_yards") >= 100) * 5)
    receiving = (col("receptions") * 1
                 + col("receiving_yards") * 0.10
                 + col("receiving_tds") * 6
                 + col("receiving_2pt_conversions") * 2
                 + (col("receiving_yards") >= 100) * 5)
    return_tds = col("special_teams_tds") * 6
    kicking = ((col("fg_made_0_19") + col("fg_made_20_29") + col("fg_made_30_39")) * 3
               + col("fg_made_40_49") * 4
               + col("fg_made_50_59") * 5
               + col("fg_made_60_") * 6
               + col("pat_made") * 1)
    return passing + rushing + receiving + return_tds + kicking


SCORING_FORMULAS = {
    "RTSPORTS": ("fantasy_points_rtsports", compute_rtsports_points),
}


# --------------------------------------------------------------------- DST (team-level)

def _points_allowed_tier(points_allowed):
    """RT Sports points-allowed scoring tiers (confirmed 2026-07-06)."""
    import numpy as np
    conditions = [points_allowed <= 2, points_allowed <= 6, points_allowed <= 13,
                  points_allowed <= 20, points_allowed <= 27]
    choices = [16.5, 13.0, 8.5, 5.0, 1.0]
    return np.select(conditions, choices, default=-2.0)


def compute_dst_points(team_week):
    """Per-row (team-game) RT Sports DST point total.

    ``team_week`` is a prepared team-perspective frame (one row per team per game — see
    ``data/team_build.py``, not this module) carrying: ``def_sacks``, ``def_interceptions``,
    ``fumble_recovery_opp``, ``fumble_recovery_tds``, ``def_tds``, ``def_safeties`` (all
    straight from nflverse ``team_stats``, verified against nflreadr's actual data dictionary —
    see the field-mapping table in the implementation plan, not guessed), plus two columns
    ``team_build.py`` derives before calling this: ``points_allowed`` (the opponent's score that
    game, from ``schedules`` — not in ``team_stats`` at all) and ``blocked_fg_defense`` /
    ``blocked_xp_defense`` (this team's defensive blocks = the *opponent's* ``fg_blocked`` /
    ``pat_blocked`` that same game, since ``team_stats`` only tracks blocks suffered by a team's
    own kicker, not blocks made by its defense).

    Two RT Sports categories are **not computable** from available nflverse data and are
    documented gaps, not silent guesses: **Blocked Punts** (``team_stats`` has no punting fields
    at all — would need play-by-play, which is a heavy/skipped-by-default dataset) and
    **Defensive/ST TD** i.e. a blocked-kick returned for a score (no field distinguishes it from
    an INT-return TD within nflverse's ``def_tds``). Both are rare (a handful league-wide per
    season) — omitted rather than approximated.

    **Int Return TD is an approximation**: nflverse's ``def_tds`` ("defensive touchdowns scored")
    isn't split by turnover type, so ``int_return_tds ≈ def_tds − fumble_recovery_tds``. Validate
    against a few known real pick-six games before trusting this for a live draft.
    """
    def col(name):
        return team_week[name] if name in team_week.columns else 0

    sacks = col("def_sacks") * 1
    interceptions = col("def_interceptions") * 2
    fumbles_recovered = col("fumble_recovery_opp") * 2
    safeties = col("def_safeties") * 2
    fumble_return_td = col("fumble_recovery_tds") * 6
    int_return_td = (col("def_tds") - col("fumble_recovery_tds")).clip(lower=0) * 6
    blocked_kicks = (col("blocked_fg_defense") + col("blocked_xp_defense")) * 2
    points_allowed = _points_allowed_tier(col("points_allowed"))
    return (sacks + interceptions + fumbles_recovered + safeties + fumble_return_td
            + int_return_td + blocked_kicks + points_allowed)


def compute_points(weekly, scoring: str):
    """Return ``(column_name, series)`` for a custom ``scoring`` formula, or ``None`` for
    ``"PPR"``/anything else — callers fall back to nflverse's ``fantasy_points_ppr``."""
    entry = SCORING_FORMULAS.get(scoring.upper())
    if entry is None:
        return None
    col_name, fn = entry
    return col_name, fn(weekly)
