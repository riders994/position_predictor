"""Tests for league configs — the shipped YAMLs and the validation that guards them.

A mistyped roster produces a plausible-looking but wrong board, which is worse than a crash, so
every validation path is covered here.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from position_predictor.eval.league import (  # noqa: E402
    LeagueConfig,
    league_from_dict,
    load_league,
    load_leagues,
)

SHIPPED = ["config/leagues/ppr_1qb.yaml", "config/leagues/suz_1qb.yaml",
           "config/leagues/my_2qb.yaml", "config/leagues/underdog_bestball.yaml"]


def _base(**over):
    return {"name": "t", "scoring": "ppr", "teams": 12,
            "starters": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1}, **over}


# -- the shipped leagues ------------------------------------------------------------------

def test_all_shipped_leagues_load():
    leagues = load_leagues(SHIPPED)
    assert [lg.name for lg in leagues] == ["ppr_1qb", "suz_1qb", "my_2qb", "underdog_bestball"]


def test_ppr_1qb_is_the_historical_default_shape():
    lg = load_league("config/leagues/ppr_1qb.yaml")
    assert (lg.scoring, lg.teams) == ("ppr", 12)
    assert lg.starters["QB"] == 1 and lg.starters["WR"] == 2
    assert lg.flex_positions == ("RB", "WR", "TE")
    assert not lg.bestball


def test_every_shipped_league_has_a_te_eligible_flex():
    """All of the user's leagues run a TE-eligible flex; regressing this quietly changes TE
    replacement level and so every TE's value."""
    for lg in load_leagues(SHIPPED):
        assert "TE" in lg.flex_positions, lg.name


def test_flex_eligibility_defaults_to_te_eligible():
    assert league_from_dict(_base()).flex_positions == ("RB", "WR", "TE")


def test_rb_wr_only_flex_can_still_be_requested():
    assert league_from_dict(_base(flex_positions=["RB", "WR"])).flex_positions == ("RB", "WR")


def test_suz_1qb_league():
    """14-team 1QB PPR with a single dedicated RB slot and two flexes. The file began as a copy
    of my_2qb.yaml, so pin the fields that copy got wrong: a stale ``name`` would collide with
    my_2qb and a stale ``scoring`` would board the league in the wrong format."""
    lg = load_league("config/leagues/suz_1qb.yaml")
    assert (lg.scoring, lg.teams, lg.starters["QB"]) == ("ppr", 14, 1)
    assert (lg.starters["RB"], lg.starters["FLEX"]) == (1, 3)
    assert not lg.bestball
    assert lg.total_picks == 210


def test_my_2qb_league():
    lg = load_league("config/leagues/my_2qb.yaml")
    assert (lg.scoring, lg.teams, lg.starters["QB"]) == ("half_ppr", 10, 2)


def test_underdog_shape():
    lg = load_league("config/leagues/underdog_bestball.yaml")
    assert (lg.scoring, lg.teams, lg.roster_size) == ("half_ppr", 12, 18)
    assert lg.starters["WR"] == 3
    assert "TE" in lg.flex_positions          # Underdog's flex takes a TE
    assert lg.bestball
    assert lg.total_picks == 216


# -- derived helpers ----------------------------------------------------------------------

def test_slot_summary_reads_like_a_league_description():
    assert league_from_dict(_base()).slot_summary() == "1QB / 2RB / 2WR / 1TE / 1FLEX"


def test_total_picks():
    assert league_from_dict(_base(teams=10, roster_size=15)).total_picks == 150


def test_missing_positions_default_to_zero_slots():
    lg = league_from_dict(_base(starters={"RB": 2, "WR": 2}))
    assert lg.starters["QB"] == 0 and lg.starters["TE"] == 0 and lg.starters["FLEX"] == 0


def test_frozen_dataclass():
    lg = league_from_dict(_base())
    assert isinstance(lg, LeagueConfig)
    with pytest.raises(Exception):
        lg.teams = 14


# -- validation ---------------------------------------------------------------------------

def test_unknown_scoring_rejected():
    with pytest.raises(ValueError, match="unknown scoring format"):
        league_from_dict(_base(scoring="ppr_but_6pt_td"))


def test_missing_teams_rejected():
    data = _base()
    del data["teams"]
    with pytest.raises(ValueError, match="missing required key 'teams'"):
        league_from_dict(data)


@pytest.mark.parametrize("teams", [2, 25])
def test_absurd_team_counts_rejected(teams):
    with pytest.raises(ValueError, match="teams must be between"):
        league_from_dict(_base(teams=teams))


def test_unknown_starter_slot_rejected():
    with pytest.raises(ValueError, match="unknown starter slot"):
        league_from_dict(_base(starters={"QB": 1, "K": 1}))


def test_empty_starters_rejected():
    with pytest.raises(ValueError, match="non-empty mapping"):
        league_from_dict(_base(starters={}))


def test_no_dedicated_slots_rejected():
    with pytest.raises(ValueError, match="no dedicated starter slots"):
        league_from_dict(_base(starters={"FLEX": 2}))


def test_unmodeled_flex_position_rejected():
    with pytest.raises(ValueError, match="not modeled"):
        league_from_dict(_base(flex_positions=["RB", "K"]))


def test_flex_slots_without_eligible_positions_rejected():
    with pytest.raises(ValueError, match="nobody can fill them"):
        league_from_dict(_base(flex_positions=[]))


def test_roster_smaller_than_lineup_rejected():
    with pytest.raises(ValueError, match="smaller than"):
        league_from_dict(_base(roster_size=3))


def test_duplicate_league_names_rejected():
    with pytest.raises(ValueError, match="duplicate league name"):
        load_leagues(["config/leagues/my_2qb.yaml", "config/leagues/my_2qb.yaml"])


def test_top_n_override_must_name_modeled_positions():
    with pytest.raises(ValueError, match="not modeled positions"):
        league_from_dict(_base(top_n={"K": 5}))


# -- QB shape / market board ----------------------------------------------------------------

def test_qb_starters_counts_dedicated_slots():
    lg = league_from_dict({"name": "x", "scoring": "ppr", "teams": 10,
                           "starters": {"QB": 2, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1}})
    assert lg.qb_starters == 2 and lg.is_superflex


def test_qb_starters_counts_a_qb_eligible_flex():
    """Superflex spells the second QB as a flex slot; it is still a two-QB league."""
    lg = league_from_dict({"name": "x", "scoring": "ppr", "teams": 12,
                           "starters": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1},
                           "flex_positions": ["QB", "RB", "WR", "TE"]})
    assert lg.qb_starters == 2 and lg.is_superflex


def test_single_qb_league_is_not_superflex():
    lg = league_from_dict({"name": "x", "scoring": "ppr", "teams": 12,
                           "starters": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1},
                           "flex_positions": ["RB", "WR", "TE"]})
    assert lg.qb_starters == 1 and not lg.is_superflex

