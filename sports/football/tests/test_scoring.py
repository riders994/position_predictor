"""Tests for scoring formats and artifact namespacing.

The load-bearing property here is that **full PPR is unchanged**: scoring became configurable
without moving a single number on the board the project has always produced.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from position_predictor.data.build import aggregate_player_seasons  # noqa: E402
from position_predictor.scoring import (  # noqa: E402
    RECEPTION_POINTS,
    normalize_scoring,
    reception_points,
    scoring_of,
)
from position_predictor.utils.config import Config  # noqa: E402
from position_predictor.utils.naming import artifact_stem  # noqa: E402


def _weekly():
    """Two players x two weeks. A is reception-heavy, B is a touchdown-dependent runner."""
    rows = [
        # player, week, receptions, standard points
        ("A", 1, 8, 10.0), ("A", 2, 6, 8.0),
        ("B", 1, 0, 12.0), ("B", 2, 1, 14.0),
    ]
    return pd.DataFrame([
        dict(player_id=p, season=2024, week=w, season_type="REG", position="RB",
             position_group="RB", receptions=r, fantasy_points=fp,
             fantasy_points_ppr=fp + r, carries=10)
        for p, w, r, fp in rows])


def _config(scoring=None):
    data = {"experiment": {"sport": "football", "position": "RB"},
            "data": {"earliest_season": 1999, "latest_completed_season": 2024},
            "target": {"predict_horizon": 1}}
    if scoring is not None:
        data["target"]["scoring"] = scoring
    return Config(data)


# -- scoring names ------------------------------------------------------------------------

@pytest.mark.parametrize("given,expected", [
    ("PPR", "ppr"), ("ppr", "ppr"), ("full_ppr", "ppr"),
    ("Half-PPR", "half_ppr"), ("half", "half_ppr"), ("0.5 ppr", "half_ppr"),
    ("standard", "standard"), ("STD", "standard"), (None, "ppr"),
])
def test_normalize_scoring_accepts_the_spellings_people_write(given, expected):
    assert normalize_scoring(given) == expected


def test_unknown_scoring_raises_rather_than_guessing():
    with pytest.raises(ValueError, match="unknown scoring format"):
        normalize_scoring("6pt_td_ppr")


def test_reception_points_table():
    assert reception_points("ppr") == 1.0
    assert reception_points("half_ppr") == 0.5
    assert reception_points("standard") == 0.0
    assert set(RECEPTION_POINTS) == {"ppr", "half_ppr", "standard"}


def test_scoring_of_defaults_to_ppr():
    assert scoring_of(_config()) == "ppr"
    assert scoring_of(_config("Half PPR")) == "half_ppr"


# -- the points formula -------------------------------------------------------------------

def test_ppr_reproduces_nflverse_ppr_exactly():
    """The whole backward-compatibility argument: under PPR, fpts == the nflverse PPR total."""
    out = aggregate_player_seasons(_weekly(), scoring="ppr")
    assert (out["fpts"] == out["fantasy_points_ppr"]).all()
    assert (out["fpts"] == out["ppr_points"]).all()


def test_half_ppr_is_the_midpoint_of_standard_and_ppr():
    out = aggregate_player_seasons(_weekly(), scoring="half_ppr")
    expected = (out["fantasy_points"] + out["fantasy_points_ppr"]) / 2
    assert out["fpts"].round(6).equals(expected.round(6))


def test_standard_drops_receptions_entirely():
    out = aggregate_player_seasons(_weekly(), scoring="standard")
    assert (out["fpts"] == out["fantasy_points"]).all()


def test_scoring_changes_who_ranks_first():
    """A (14 rec, 18 std pts) beats B (1 rec, 26 std pts) in PPR but loses in standard —
    exactly the reordering half-PPR exists to capture."""
    ppr = aggregate_player_seasons(_weekly(), scoring="ppr").set_index("player_id")["ppg"]
    std = aggregate_player_seasons(_weekly(), scoring="standard").set_index("player_id")["ppg"]
    half = aggregate_player_seasons(_weekly(), scoring="half_ppr").set_index("player_id")["ppg"]
    assert ppr["A"] > ppr["B"]
    assert std["A"] < std["B"]
    # half-PPR sits between the two for every player
    for p in ("A", "B"):
        assert std[p] < half[p] < ppr[p]


def test_ppr_points_stays_true_ppr_under_any_scoring():
    """`ppr_points` feeds features (finish_ppr_rank, ppr_per_touch) and eligibility cutoffs, so
    its meaning must not drift with a league setting."""
    for scoring in RECEPTION_POINTS:
        out = aggregate_player_seasons(_weekly(), scoring=scoring)
        assert (out["ppr_points"] == out["fantasy_points_ppr"]).all()


def test_ppg_divides_by_games():
    out = aggregate_player_seasons(_weekly(), scoring="half_ppr").set_index("player_id")
    assert out.loc["A", "games"] == 2
    assert out.loc["A", "ppg"] == pytest.approx(out.loc["A", "fpts"] / 2)


# -- artifact namespacing -----------------------------------------------------------------

def test_ppr_keeps_the_historical_stem():
    assert artifact_stem(_config()) == "football_rb"
    assert artifact_stem(_config("PPR")) == "football_rb"


def test_alternate_scoring_gets_its_own_stem():
    assert artifact_stem(_config("half_ppr")) == "football_rb_half_ppr"
    assert artifact_stem(_config("standard")) == "football_rb_standard"


def test_stem_accepts_an_explicit_position():
    assert artifact_stem(_config("half_ppr"), position="WR") == "football_wr_half_ppr"
