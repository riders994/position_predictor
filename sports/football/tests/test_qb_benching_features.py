"""Tests for the stage-2 preseason feature layer (pure frame logic; no nflverse / network).

These do not check that any feature is predictive. They check the one property the whole stage
depends on — **every column is knowable by Sept 1 of the season it describes** — plus the join
defects that have already bitten this project once.

The leakage tests are deliberately built so that season N and season N-1 carry obviously
different values, and assert the feature carries the earlier one.
"""

import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qb_benching.features import (  # noqa: E402
    add_history, add_prior_play, add_room, add_team, add_tenure,
)

QB, OTHER = "00-0000001", "00-0000002"


def _cohort(rows):
    return pl.DataFrame(rows, schema_overrides={"season": pl.Int32})


WEEKLY_SCHEMA = {
    "player_id": pl.String, "season": pl.Int32, "week": pl.Int32, "season_type": pl.String,
    "position": pl.String, "attempts": pl.Int64, "completions": pl.Int64,
    "passing_yards": pl.Int64, "passing_tds": pl.Int64, "interceptions": pl.Int64,
    "sacks": pl.Int64, "passing_epa": pl.Float64, "passing_cpoe": pl.Float64,
    "carries": pl.Int64, "rushing_yards": pl.Int64, "rushing_tds": pl.Int64,
}


def _weekly(rows):
    """Always fully schema'd — an empty frame with no columns fails inside the block, not here."""
    base = {"season_type": "REG", "position": "QB", "attempts": 0, "completions": 0,
            "passing_yards": 0, "passing_tds": 0, "interceptions": 0, "sacks": 0,
            "passing_epa": 0.0, "passing_cpoe": 0.0, "carries": 0, "rushing_yards": 0,
            "rushing_tds": 0}
    return pl.DataFrame([{**base, **r} for r in rows], schema=WEEKLY_SCHEMA)


STARTERS_SCHEMA = {"season": pl.Int32, "week": pl.Int32, "team": pl.String,
                   "starter_id": pl.String}


def _starters(rows):
    return pl.DataFrame(rows, schema=STARTERS_SCHEMA)


# -- leakage: features must describe season N-1 ------------------------------------------------

def test_prior_play_reads_the_previous_season_not_the_current_one():
    """The 2021 row must carry 2020's volume. Reading N would be a straight label leak."""
    cohort = _cohort([{"season": 2021, "team": "SF", "opener_id": QB, "opener_name": "A"}])
    weekly = _weekly([
        {"player_id": QB, "season": 2020, "week": 1, "attempts": 100},
        {"player_id": QB, "season": 2021, "week": 1, "attempts": 999},
    ])
    starters = _starters([{"season": 2020, "week": 1, "team": "SF", "starter_id": QB}])
    out, cols = add_prior_play(cohort, weekly, starters)
    assert out["prior_attempts"][0] == 100
    assert out["prior_starts"][0] == 1
    assert "prior_epa_per_db" in cols


def test_an_opener_with_no_prior_season_is_flagged_not_zeroed_silently():
    cohort = _cohort([{"season": 2021, "team": "SF", "opener_id": QB, "opener_name": "A"}])
    weekly = _weekly([{"player_id": QB, "season": 2021, "week": 1, "attempts": 500}])
    out, _ = add_prior_play(cohort, weekly, _starters([]))
    assert out["has_prior_season"][0] is False
    assert out["prior_attempts"][0] == 0        # volume: "did not play" is the fact
    assert out["prior_epa_per_db"][0] is None   # rates stay null rather than inventing a zero


def test_history_lags_the_label_by_exactly_one_season():
    """`benched_last_season` on the 2021 row is the 2020 row's own outcome."""
    cohort = _cohort([
        {"season": 2020, "team": "SF", "opener_id": QB, "opener_name": "A",
         "benched": True, "displaced": True, "n_benched": 6},
        {"season": 2021, "team": "SF", "opener_id": QB, "opener_name": "A",
         "benched": False, "displaced": False, "n_benched": 0},
    ])
    frame, _ = add_history(cohort)
    row21 = frame.filter(pl.col("season") == 2021).row(0, named=True)
    row20 = frame.filter(pl.col("season") == 2020).row(0, named=True)
    assert row21["benched_last_season"] is True and row21["opened_last_season"] is True
    assert row20["benched_last_season"] is False and row20["opened_last_season"] is False


def test_history_never_reads_its_own_season():
    """A single-season cohort must produce no history at all — there is nothing before it."""
    cohort = _cohort([{"season": 2021, "team": "SF", "opener_id": QB, "opener_name": "A",
                       "benched": True, "displaced": True, "n_benched": 9}])
    frame, _ = add_history(cohort)
    assert frame["benched_last_season"][0] is False
    assert frame["opened_last_season"][0] is False


# -- the join defects --------------------------------------------------------------------------

def test_a_relocated_franchise_keeps_its_prior_record():
    """The cohort is canonicalised (LA) and schedules is not (STL); the join must still land."""
    cohort = _cohort([{"season": 2016, "team": "LA", "opener_id": QB, "opener_name": "A"}])
    schedules = pl.DataFrame([
        {"season": 2015, "game_type": "REG", "home_team": "STL", "away_team": "SF",
         "home_score": 24, "away_score": 10},
    ], schema_overrides={"season": pl.Int32})
    coach = pl.DataFrame([{"season": 2015, "team": "LA", "coach": "Fisher"}],
                         schema_overrides={"season": pl.Int32})
    out, _ = add_team(cohort, schedules, coach)
    assert out["prior_win_pct"][0] == 1.0
    assert out["prior_point_diff"][0] == 14.0


def test_new_head_coach_is_a_change_not_a_missing_value():
    cohort = _cohort([{"season": 2021, "team": "SF", "opener_id": QB, "opener_name": "A"},
                      {"season": 2022, "team": "SF", "opener_id": QB, "opener_name": "A"}])
    schedules = pl.DataFrame([], schema={"season": pl.Int32, "game_type": pl.String,
                                         "home_team": pl.String, "away_team": pl.String,
                                         "home_score": pl.Float64, "away_score": pl.Float64})
    coach = pl.DataFrame([{"season": 2020, "team": "SF", "coach": "Old"},
                          {"season": 2021, "team": "SF", "coach": "Old"},
                          {"season": 2022, "team": "SF", "coach": "New"}],
                         schema_overrides={"season": pl.Int32})
    out, _ = add_team(cohort, schedules, coach)
    got = {r["season"]: r["new_head_coach"] for r in out.iter_rows(named=True)}
    assert got[2021] is False and got[2022] is True


# -- the fill-in opener ------------------------------------------------------------------------

def test_incumbent_present_flags_the_fill_in_opener():
    """The Derek Anderson shape: last year's starter is on the week-1 chart but is not the opener.

    Losing the job back to him is not a benching on the merits, so the model must see it.
    """
    cohort = _cohort([{"season": 2021, "team": "SF", "opener_id": QB, "opener_name": "A"}])
    depth = pl.DataFrame([
        {"season": 2021, "week": 1, "team": "SF", "opener_id": QB, "depth_rank": 1},
        {"season": 2021, "week": 1, "team": "SF", "opener_id": OTHER, "depth_rank": 2},
    ], schema_overrides={"season": pl.Int32, "week": pl.Int32, "depth_rank": pl.Int32})
    starters = _starters([{"season": 2020, "week": w, "team": "SF", "starter_id": OTHER}
                          for w in range(1, 6)])
    draft = pl.DataFrame([], schema={"season": pl.Int32, "team": pl.String, "position": pl.String,
                                     "pick": pl.Float64})
    out, cols = add_room(cohort, depth, _weekly([]), draft, starters)
    assert out["incumbent_present"][0] is True
    assert out["room_size"][0] == 2
    assert "backup_prior_epa_per_db" in cols


def test_incumbent_present_is_false_when_the_opener_is_the_incumbent():
    cohort = _cohort([{"season": 2021, "team": "SF", "opener_id": QB, "opener_name": "A"}])
    depth = pl.DataFrame([
        {"season": 2021, "week": 1, "team": "SF", "opener_id": QB, "depth_rank": 1},
    ], schema_overrides={"season": pl.Int32, "week": pl.Int32, "depth_rank": pl.Int32})
    starters = _starters([{"season": 2020, "week": w, "team": "SF", "starter_id": QB}
                          for w in range(1, 6)])
    draft = pl.DataFrame([], schema={"season": pl.Int32, "team": pl.String, "position": pl.String,
                                     "pick": pl.Float64})
    out, _ = add_room(cohort, depth, _weekly([]), draft, starters)
    assert out["incumbent_present"][0] is False


def test_room_uses_the_week_one_chart_only():
    """A week-8 demotion must not leak backwards into the preseason room."""
    cohort = _cohort([{"season": 2021, "team": "SF", "opener_id": QB, "opener_name": "A"}])
    depth = pl.DataFrame([
        {"season": 2021, "week": 1, "team": "SF", "opener_id": QB, "depth_rank": 1},
        {"season": 2021, "week": 8, "team": "SF", "opener_id": QB, "depth_rank": 2},
        {"season": 2021, "week": 8, "team": "SF", "opener_id": OTHER, "depth_rank": 1},
    ], schema_overrides={"season": pl.Int32, "week": pl.Int32, "depth_rank": pl.Int32})
    draft = pl.DataFrame([], schema={"season": pl.Int32, "team": pl.String, "position": pl.String,
                                     "pick": pl.Float64})
    out, _ = add_room(cohort, depth, _weekly([]), draft, _starters([]))
    assert out["room_size"][0] == 1     # week 1 had one quarterback on the chart, not three


def test_tenure_marks_the_undrafted_rather_than_imputing_a_pick():
    cohort = _cohort([{"season": 2021, "team": "SF", "opener_id": QB, "opener_name": "A"}])
    rosters = pl.DataFrame([{"season": 2021, "player_id": QB, "years_exp": 3.0,
                             "birth_date": "1995-01-01"}], schema_overrides={"season": pl.Int32})
    draft = pl.DataFrame([], schema={"gsis_id": pl.String, "pick": pl.Float64,
                                     "round": pl.Float64})
    out, _ = add_tenure(cohort, rosters, draft)
    assert out["was_drafted"][0] is False
    assert out["draft_pick"][0] is None
    assert out["age"][0] == 26.0
