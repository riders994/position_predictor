"""Tests for the high-school recruiting layer: parsing, cleaning, and the cohort join.

The join is the dangerous part. At 17 positives, one wrong name match is enough to manufacture a
finding, so these lock in the conservative behaviour: match within a plausible class window, and
refuse ambiguity rather than guess.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qb_breakout.data.link import link_recruits_to_cohort, normalize_name  # noqa: E402
from qb_breakout.data.recruiting import clean_measurables, parse_recruit  # noqa: E402


def _item(name="Kyler Murray", pos="QB-DT", cls=2015, grade=90, attrs=None, schools=None):
    return {
        "recruitingClass": cls,
        "status": {"description": "Signed"},
        "grade": grade,
        "athlete": {
            "id": "1", "alternateId": "9", "fullName": name,
            "firstName": name.split()[0], "lastName": name.split()[-1],
            "height": 71.0, "weight": 178.0,
            "position": {"abbreviation": pos},
            "highSchool": {"properName": "Allen High School",
                           "address": {"city": "Allen", "stateAbbreviation": "TX"}},
            "hometown": {"city": "Allen", "stateAbbreviation": "TX"},
        },
        "attributes": attrs if attrs is not None else [
            {"name": "rank", "value": 13.0},
            {"name": "positionRank", "value": 1.0},
            {"name": "fortyYrdDash", "value": 4.5},
        ],
        "schools": schools if schools is not None else [
            {"visit": "2015-01-16T08:00Z"}, {"visit": "2014-10-17T07:00Z"}, {},
        ],
    }


# ------------------------------------------------------------------------------------ parsing


def test_parse_extracts_profile_ranks_and_high_school():
    rec = parse_recruit(_item())
    assert rec["recruit_name"] == "Kyler Murray"
    assert rec["espn_grade"] == 90
    assert rec["rank_national"] == 13.0
    assert rec["rank_position"] == 1.0
    assert rec["hs_state"] == "TX"
    assert rec["forty_yd"] == 4.5


def test_scout_assigned_qb_archetype_is_captured():
    """ESPN's QB-DT / QB-PP split is a hindsight-free high-school archetype label."""
    assert parse_recruit(_item(pos="QB-DT"))["dual_threat"] == 1
    assert parse_recruit(_item(pos="QB-PP"))["dual_threat"] == 0
    ath = parse_recruit(_item(pos="ATH"))
    assert ath["recruited_as_athlete"] == 1
    assert ath["dual_threat"] == 0


def test_recruitment_shape_counts_schools_and_visits():
    rec = parse_recruit(_item())
    assert rec["n_schools_involved"] == 3
    assert rec["n_official_visits"] == 2
    assert rec["first_visit"].startswith("2014")


# ----------------------------------------------------------------------------------- cleaning


def test_impossible_measurables_are_nulled_not_kept():
    """ESPN reports three-cone values like 99.0; a 99-second cone drill is not a slow player."""
    df = pd.DataFrame([{"forty_yd": 4.5, "three_cone": 99.0, "shuttle_20yd": 4.4,
                        "vertical_jump": 30.0}])
    out = clean_measurables(df)
    assert out.loc[0, "three_cone"] is None or pd.isna(out.loc[0, "three_cone"])
    assert out.loc[0, "forty_yd"] == 4.5
    assert out.loc[0, "n_measurables"] == 3


# ------------------------------------------------------------------------------ name handling


def test_name_normalisation_bridges_the_two_sources():
    assert normalize_name("Michael Penix Jr.") == normalize_name("Michael Penix")
    assert normalize_name("E.J. Manuel") == normalize_name("EJ Manuel")
    assert normalize_name("Robert Griffin III") == normalize_name("Robert Griffin")
    assert normalize_name(None) == ""


# ------------------------------------------------------------------------------------- joining


def _careers(rows):
    return pd.DataFrame(rows)


def _recruits(rows):
    return pd.DataFrame(rows)


def test_unique_in_window_match_attaches_the_profile():
    careers = _careers([dict(player_id="A", player_name="Kyler Murray", entry_season=2019)])
    recruits = _recruits([dict(recruit_name="Kyler Murray", recruit_class=2015, espn_grade=90)])
    out = link_recruits_to_cohort(careers, recruits)
    assert out.loc[0, "matched"]
    assert out.loc[0, "match_reason"] == "unique"
    assert out.loc[0, "espn_grade"] == 90


def test_name_match_outside_the_plausible_class_window_is_rejected():
    """A recruit signing the same year the QB entered the NFL is a different person."""
    careers = _careers([dict(player_id="A", player_name="John Smith", entry_season=2019)])
    recruits = _recruits([dict(recruit_name="John Smith", recruit_class=2019, espn_grade=70)])
    out = link_recruits_to_cohort(careers, recruits)
    assert not out.loc[0, "matched"]
    assert out.loc[0, "match_reason"] == "name_match_out_of_window"


def test_two_candidates_resolve_to_the_modal_five_year_lag():
    careers = _careers([dict(player_id="A", player_name="John Smith", entry_season=2019)])
    recruits = _recruits([
        dict(recruit_name="John Smith", recruit_class=2014, espn_grade=80),  # lag 5
        dict(recruit_name="John Smith", recruit_class=2016, espn_grade=60),  # lag 3
    ])
    out = link_recruits_to_cohort(careers, recruits)
    assert out.loc[0, "match_reason"] == "modal_lag"
    assert out.loc[0, "espn_grade"] == 80


def test_genuinely_ambiguous_candidates_are_refused_not_guessed():
    """Equidistant same-named prospects stay unmatched — a coin flip here invents data."""
    careers = _careers([dict(player_id="A", player_name="John Smith", entry_season=2019)])
    recruits = _recruits([
        dict(recruit_name="John Smith", recruit_class=2013, espn_grade=80),  # lag 6
        dict(recruit_name="John Smith", recruit_class=2015, espn_grade=60),  # lag 4
    ])
    out = link_recruits_to_cohort(careers, recruits)
    assert not out.loc[0, "matched"]
    assert out.loc[0, "match_reason"] == "ambiguous"
    assert out.loc[0, "n_candidates"] == 2


def test_unmatched_rows_are_kept_in_the_cohort():
    """Dropping unmatched QBs would silently reshape the cohort; they stay, flagged."""
    careers = _careers([
        dict(player_id="A", player_name="Kyler Murray", entry_season=2019),
        dict(player_id="B", player_name="Nobody Here", entry_season=2019),
    ])
    recruits = _recruits([dict(recruit_name="Kyler Murray", recruit_class=2015, espn_grade=90)])
    out = link_recruits_to_cohort(careers, recruits)
    assert len(out) == 2
    assert list(out["matched"]) == [True, False]
    assert out.loc[1, "match_reason"] == "no_name_match"
