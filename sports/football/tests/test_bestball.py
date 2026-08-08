"""Tests for best-ball valuation (pure logic; synthetic weekly scores, no model/network)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from position_predictor.eval.bestball import (  # noqa: E402
    DEFAULT_LAMBDAS,
    apply_bestball,
    projected_sigma,
    realized_bestball_ppg,
    season_volatility,
    starters_per_position,
    weekly_fantasy_points,
    weekly_thresholds,
)
from position_predictor.eval.league import league_from_dict  # noqa: E402


def _league(**over):
    """A 4-team league starting 1 of each position (the smallest the validator allows)."""
    return league_from_dict({"name": "t", "scoring": "half_ppr", "teams": 4,
                             "starters": {"QB": 1, "RB": 1, "WR": 1, "TE": 1, "FLEX": 1},
                             "flex_positions": ["RB", "WR"], "roster_size": 10, **over})


def _no_flex():
    """Same league without a flex, so a position's starter count is exactly ``teams``."""
    return _league(starters={"QB": 1, "RB": 1, "WR": 1, "TE": 1, "FLEX": 0})


def _weekly(rows):
    """rows: (player_id, season, week, position, receptions, standard_points)."""
    return pd.DataFrame([
        dict(player_id=p, season=s, week=w, season_type="REG", position=pos,
             position_group=pos, receptions=r, fantasy_points=fp)
        for p, s, w, pos, r, fp in rows])


# -- weekly points ------------------------------------------------------------------------

def test_weekly_points_apply_the_scoring_format():
    wk = _weekly([("A", 2024, 1, "RB", 4, 10.0)])
    assert weekly_fantasy_points(wk, scoring="ppr")["points"].iloc[0] == 14.0
    assert weekly_fantasy_points(wk, scoring="half_ppr")["points"].iloc[0] == 12.0
    assert weekly_fantasy_points(wk, scoring="standard")["points"].iloc[0] == 10.0


def test_weekly_points_drop_non_modeled_positions_and_postseason():
    wk = _weekly([("A", 2024, 1, "RB", 0, 10.0), ("K", 2024, 1, "K", 0, 9.0)])
    wk.loc[wk.player_id == "A", "season_type"] = "POST"
    assert weekly_fantasy_points(wk).empty


# -- starter counts & thresholds ----------------------------------------------------------

def test_flex_slots_are_split_across_eligible_positions():
    counts = starters_per_position(_league())     # 4 teams, 1 flex each -> 4 flex slots
    assert counts["QB"] == 4 and counts["TE"] == 4
    assert counts["RB"] == 6 and counts["WR"] == 6    # 4 dedicated + 4/2 flex share


def test_threshold_is_the_marginal_starters_score():
    # 6 RBs in one week; the league starts 4 -> threshold = the 4th best score.
    wk = _weekly([(f"RB{i}", 2024, 1, "RB", 0, pts)
                  for i, pts in enumerate([20.0, 15.0, 12.0, 10.0, 5.0, 1.0])])
    th = weekly_thresholds(weekly_fantasy_points(wk, scoring="standard"), _no_flex())
    assert th[th.position == "RB"]["threshold"].iloc[0] == 10.0


# -- realized best-ball value -------------------------------------------------------------

def test_bad_weeks_cost_nothing_but_good_weeks_count():
    """The defining best-ball property: below-threshold weeks floor at zero, not negative."""
    scores = [20.0, 15.0, 12.0, 10.0, 5.0, 1.0]      # threshold = 4th best = 10.0
    wk = _weekly([(f"RB{i}", 2024, w, "RB", 0, pts)
                  for w in (1, 2) for i, pts in enumerate(scores)])
    act = realized_bestball_ppg(weekly_fantasy_points(wk, scoring="standard"), _no_flex())
    act = act.set_index("player_id")["bestball_ppg_actual"]
    assert act["RB0"] == pytest.approx(10.0)     # (20-10) each week
    assert act["RB3"] == pytest.approx(0.0)      # exactly at the threshold
    assert act["RB5"] == pytest.approx(0.0)      # far below -> zero, never negative


def test_missed_weeks_are_averaged_over_league_weeks_not_games_played():
    """An injury costs a best-ball roster real points — you cannot replace the player."""
    full = [("A", 2024, w, "RB", 0, 30.0) for w in (1, 2)]
    half = [("B", 2024, 1, "RB", 0, 30.0)]                    # plays week 1 only
    filler = [(f"F{i}", 2024, w, "RB", 0, 1.0) for i in range(4) for w in (1, 2)]
    act = realized_bestball_ppg(
        weekly_fantasy_points(_weekly(full + half + filler), scoring="standard"), _no_flex())
    act = act.set_index("player_id")["bestball_ppg_actual"]
    assert act["A"] == pytest.approx(29.0)                    # (30-1) in both league weeks
    assert act["B"] == pytest.approx(act["A"] / 2)            # half the weeks, half the value


# -- volatility ---------------------------------------------------------------------------

def test_season_volatility_needs_a_minimum_of_weeks():
    wk = _weekly([("A", 2024, w, "RB", 0, 10.0 + w) for w in range(1, 6)] +
                 [("B", 2024, w, "RB", 0, 10.0) for w in (1, 2)])
    vol = season_volatility(weekly_fantasy_points(wk, scoring="standard"))
    assert set(vol["player_id"]) == {"A"}     # B has 2 games, below the 4-game floor


def test_projected_sigma_shrinks_short_samples_toward_the_position_mean():
    """A player with few games sits closer to the position mean than his own noisy sample."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(10):                       # a stable population with sigma ~= 3
        for w in range(1, 17):
            rows.append((f"P{i}", 2023, w, "RB", 0, float(10 + rng.normal(0, 3))))
    for w in range(1, 6):                     # one wild 5-game player
        rows.append(("WILD", 2023, w, "RB", 0, float(10 + rng.normal(0, 15))))
    vol = season_volatility(weekly_fantasy_points(_weekly(rows), scoring="standard"))
    sig = projected_sigma(vol, board_season=2024).set_index("player_id")
    raw_wild = vol.set_index("player_id").loc["WILD", "week_sd"]
    assert sig.loc["WILD", "sigma"] < raw_wild        # shrunk down toward the pool
    assert sig.loc["WILD", "sigma"] > sig.loc["P0", "sigma"]   # but still the most volatile


def test_projected_sigma_only_uses_seasons_before_the_board():
    rows = [("A", s, w, "RB", 0, 10.0 + w) for s in (2023, 2024) for w in range(1, 10)]
    vol = season_volatility(weekly_fantasy_points(_weekly(rows), scoring="standard"))
    assert projected_sigma(vol, board_season=2023).empty      # nothing precedes 2023
    assert not projected_sigma(vol, board_season=2024).empty


# -- the board adjustment -----------------------------------------------------------------

def _proj():
    return pd.DataFrame({"player_id": ["A", "B"], "player_name": ["a", "b"],
                         "position": ["RB", "RB"], "proj_ppg": [12.0, 12.0],
                         "proj_pos_rank": [1, 2]})


def test_default_lambdas_are_zero_so_bestball_equals_the_mean_board():
    """The fitted upside weight is zero (see the module docstring) — the default must not tilt."""
    assert set(DEFAULT_LAMBDAS.values()) == {0.0}
    sigma = pd.DataFrame({"player_id": ["A", "B"], "position": ["RB", "RB"],
                          "sigma": [2.0, 8.0], "sigma_games": [16, 16]})
    out = apply_bestball(_proj(), sigma, DEFAULT_LAMBDAS)
    assert (out["bestball_ppg"] == out["proj_ppg"]).all()


def test_explicit_lambda_rewards_the_more_volatile_player():
    sigma = pd.DataFrame({"player_id": ["A", "B"], "position": ["RB", "RB"],
                          "sigma": [2.0, 8.0], "sigma_games": [16, 16]})
    out = apply_bestball(_proj(), sigma, {"RB": 0.5}).set_index("player_id")
    assert out.loc["A", "bestball_ppg"] == 13.0
    assert out.loc["B", "bestball_ppg"] == 16.0


def test_players_without_history_fall_back_to_the_position_mean():
    sigma = pd.DataFrame({"player_id": ["A"], "position": ["RB"], "sigma": [4.0],
                          "sigma_games": [16]})
    out = apply_bestball(_proj(), sigma, {"RB": 0.5}).set_index("player_id")
    assert out.loc["B", "sigma"] == 4.0        # B is neither rewarded nor punished
    assert out.loc["B", "bestball_ppg"] == out.loc["A", "bestball_ppg"]
