"""Tests for the QB-benching label layer (pure frame logic; no nflverse / network).

None of these check that any particular quarterback was benched. They check that the label
cannot be fooled by the three data defects that actually bit while it was being built:

* ``rosters_weekly.status`` is a season-final stamp before 2021, so a quarterback who ends the
  year on injured reserve reads ``RES`` in the weeks he was starting. The first version of this
  label used it and reported zero benchings in 2015.
* ``schedules`` and ``rosters_weekly`` spell St. Louis differently (``STL`` vs ``SL``), which
  read every Rams quarterback as playing for another club.
* Injured reserve is invisible to the injury report, so "not listed hurt" does not mean healthy.

Plus the arithmetic: every displaced week must be classified exactly once.
"""

import sys
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qb_benching.labels import (  # noqa: E402
    attach_depth, build_cohort, classify_displacement, displacement_panel, game_starters,
    opening_starters, qb_depth, reconciles,
)
from qb_benching.labels.cohort import sensitivity  # noqa: E402

STARTER, BACKUP = "00-0000001", "00-0000002"


def _schedules(weeks=range(1, 6), team="SF", opp="SEA", starters=None):
    """A one-club season. ``starters`` maps week -> the gsis_id that started."""
    starters = starters or {}
    rows = []
    for w in weeks:
        rows.append({
            "season": 2015, "week": w, "game_type": "REG",
            "home_team": team, "away_team": opp,
            "home_qb_id": starters.get(w, STARTER), "away_qb_id": "00-0000009",
            "home_qb_name": "Starter Man", "away_qb_name": "Other Guy",
            "gameday": f"2015-09-{10 + w:02d}",
        })
    return pl.DataFrame(rows)


def _rosters(weeks, status="ACT", team="SF"):
    return pl.DataFrame([
        {"season": 2015, "week": w, "game_type": "REG", "position": "QB",
         "gsis_id": STARTER, "status": status, "team": team}
        for w in weeks
    ])


def _injuries(weeks):
    return pl.DataFrame([
        {"season": 2015, "week": w, "game_type": "REG", "position": "QB",
         "gsis_id": STARTER, "report_status": "Out"} for w in weeks
    ], schema={"season": pl.Int64, "week": pl.Int64, "game_type": pl.String,
               "position": pl.String, "gsis_id": pl.String, "report_status": pl.String})


def _depth(chart, team="SF"):
    """``chart`` maps week -> list of gsis_ids in declared order."""
    rows = []
    for w, ids in chart.items():
        for rank, pid in enumerate(ids, start=1):
            rows.append({"season": 2015, "week": w, "game_type": "REG", "club_code": team,
                         "gsis_id": pid, "depth_position": "QB", "depth_team": str(rank)})
    return pl.DataFrame(rows)


def _focus(frame, team="SF"):
    """Only the club under test. Every club in the schedule gets an opener row, including the
    fixed opponent, and an assertion that reads row 0 would read whichever sorts first."""
    return frame.filter(pl.col("team") == team)


def _build(schedules, rosters, injuries, depth=None):
    starters = game_starters(schedules)
    panel = displacement_panel(starters, opening_starters(starters), rosters, injuries)
    if depth is not None:
        panel = attach_depth(panel, qb_depth(depth))
    return classify_displacement(panel)


# -- the season-final status trap -------------------------------------------------------------

def test_pre_2021_reserve_stamp_does_not_erase_a_benching():
    """The 2015 Kaepernick shape: benched from week 4, and the whole season stamped ``RES``.

    Reading ``status`` as a weekly value calls every one of these weeks an injury and the
    benching disappears. Presence — and the depth chart — must win.
    """
    sched = _schedules(range(1, 6), starters={4: BACKUP, 5: BACKUP})
    rosters = _rosters(range(1, 6), status="RES")   # the season-final stamp, on every week
    depth = _depth({w: ([STARTER, BACKUP] if w < 4 else [BACKUP, STARTER]) for w in range(1, 6)})
    out = _build(sched, rosters, _injuries([]), depth)
    displaced = out.filter(~pl.col("held"))
    assert displaced.height == 2
    assert set(displaced["outcome"]) == {"benched"}


def test_modern_reserve_status_is_believed():
    """From 2021 the same column IS weekly, so a RES week is an injury, not a benching."""
    sched = _schedules(range(1, 6), starters={4: BACKUP, 5: BACKUP}).with_columns(
        pl.lit(2022).alias("season"))
    rosters = _rosters(range(1, 6), status="RES").with_columns(pl.lit(2022).alias("season"))
    out = _build(sched, rosters, _injuries([]).with_columns(pl.lit(2022).alias("season")))
    displaced = out.filter(~pl.col("held"))
    assert set(displaced["outcome"]) == {"injured"}


# -- the team-code trap -----------------------------------------------------------------------

def test_st_louis_spelling_is_not_read_as_another_club():
    """``schedules`` says STL and ``rosters_weekly`` says SL; both are the Rams.

    Uncanonicalised, the club comparison fires and a benching is recorded as a departure.
    """
    sched = _schedules(range(1, 6), team="STL", starters={4: BACKUP, 5: BACKUP})
    rosters = _rosters(range(1, 6), team="SL")
    out = _build(sched, rosters, _injuries([]))
    displaced = out.filter(~pl.col("held"))
    assert "gone" not in set(displaced["outcome"])
    assert set(displaced["outcome"]) == {"benched"}


# -- injured reserve is invisible to the report -----------------------------------------------

def test_dropping_off_the_depth_chart_is_an_injury_not_a_benching():
    """A quarterback on IR leaves the report entirely, so absence of a listing proves nothing.

    Falling off his club's chart is what marks him unavailable.
    """
    sched = _schedules(range(1, 6), starters={4: BACKUP, 5: BACKUP})
    rosters = _rosters(range(1, 4))              # rows stop when he goes on reserve
    depth = _depth({1: [STARTER, BACKUP], 2: [STARTER, BACKUP], 3: [STARTER, BACKUP],
                    4: [BACKUP], 5: [BACKUP]})   # off the chart from week 4
    out = _build(sched, rosters, _injuries([]), depth)
    displaced = out.filter(~pl.col("held"))
    assert set(displaced["outcome"]) == {"injured"}
    assert set(displaced["evidence"]) == {"off_chart"}


def test_injury_report_outranks_the_chart():
    """Listed hurt and demoted in the same week counts as hurt — benching stays a lower bound."""
    sched = _schedules(range(1, 6), starters={4: BACKUP})
    depth = _depth({w: ([STARTER, BACKUP] if w < 4 else [BACKUP, STARTER]) for w in range(1, 6)})
    out = _focus(_build(sched, _rosters(range(1, 6)), _injuries([4]), depth))
    assert out.filter(pl.col("week") == 4)["outcome"][0] == "injured"


def test_an_unpublished_chart_week_is_not_evidence_of_absence():
    """A club-week with no chart at all must not read as the quarterback falling off it."""
    sched = _schedules(range(1, 6), starters={4: BACKUP})
    depth = _depth({1: [STARTER, BACKUP]})       # only week 1 was published
    out = _focus(_build(sched, _rosters(range(1, 6)), _injuries([]), depth))
    assert out.filter(pl.col("week") == 4)["evidence"][0] != "off_chart"


# -- arithmetic -------------------------------------------------------------------------------

def test_every_displaced_week_is_classified_exactly_once():
    sched = _schedules(range(1, 6), starters={3: BACKUP, 4: BACKUP, 5: BACKUP})
    out = _build(sched, _rosters(range(1, 6)), _injuries([3]))
    cohort = _focus(build_cohort(out))
    assert reconciles(cohort)
    row = cohort.row(0, named=True)
    assert row["n_held"] + row["n_benched"] + row["n_injured"] + row["n_gone"] == \
        row["games_after_opener"]


def test_the_opener_is_the_first_game_played_not_week_one():
    """2001 moved a week; a club whose season starts at week 2 still has an opening starter."""
    sched = _schedules(range(2, 6), starters={4: BACKUP})
    openers = _focus(opening_starters(game_starters(sched)))
    assert openers["opening_week"][0] == 2
    assert openers["opener_id"][0] == STARTER


def test_bye_weeks_never_become_missed_starts():
    """The grid is the club's actual games, so a skipped week is simply absent."""
    sched = _schedules([1, 2, 3, 5, 6], starters={5: BACKUP})
    out = _focus(_build(sched, _rosters([1, 2, 3, 5, 6]), _injuries([])))
    assert 4 not in set(out["week"])
    assert out.filter(~pl.col("held")).height == 1


def test_the_bar_is_monotone_in_weeks():
    sched = _schedules(range(1, 8), starters={5: BACKUP, 6: BACKUP, 7: BACKUP})
    out = _build(sched, _rosters(range(1, 8)), _injuries([]))
    sens = sensitivity(out, bars=(2, 3, 4))
    counts = sens.sort("bar")["benched"].to_list()
    assert counts == sorted(counts, reverse=True)


@pytest.mark.parametrize("bar,expected", [(2, True), (3, True), (4, False)])
def test_label_fires_at_the_declared_bar(bar, expected):
    sched = _schedules(range(1, 8), starters={5: BACKUP, 6: BACKUP, 7: BACKUP})
    out = _build(sched, _rosters(range(1, 8)), _injuries([]))
    cohort = _focus(build_cohort(out, benched_weeks=bar))
    assert bool(cohort["benched"][0]) is expected
