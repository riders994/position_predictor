"""Tests for the draft simulator (pure logic; synthetic pools, no model/network)."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from position_predictor.eval.draftsim import (  # noqa: E402
    POS_INDEX,
    AdpPolicy,
    BoardPolicy,
    DraftSim,
    LookaheadPolicy,
    Pool,
    adp_spread,
    calibrate_noise,
    fit_spread,
    lineup_value,
    position_caps,
    simulated_pick_spread,
    snake_order,
)
from position_predictor.eval.league import league_from_dict  # noqa: E402


def _league(**over):
    base = {"name": "t", "scoring": "ppr", "teams": 4,
            "starters": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1},
            "flex_positions": ["RB", "WR", "TE"], "roster_size": 8}
    return league_from_dict({**base, **over})


def _pool(rows, board_order=None):
    """rows: (position, adp, value); board order defaults to value descending over scored rows."""
    pos = [POS_INDEX[p] for p, _, _ in rows]
    adp = [a for _, a, _ in rows]
    val = [v for _, _, v in rows]
    if board_order is None:
        scored = [i for i, v in enumerate(val) if np.isfinite(v)]
        board_order = sorted(scored, key=lambda i: -val[i])
    return Pool(position=pos, adp=adp, value=val, board_order=board_order)


def test_snake_order_reverses_every_other_round():
    assert snake_order(3, 3).tolist() == [0, 1, 2, 2, 1, 0, 0, 1, 2]


def test_fit_spread_recovers_power_law():
    adp = np.arange(1, 200, dtype=float)
    a, b = fit_spread(adp, 0.4 * adp ** 0.7)
    assert a == pytest.approx(0.4, rel=1e-6)
    assert b == pytest.approx(0.7, rel=1e-6)
    assert adp_spread([1, 100], a=0.4, b=0.7) == pytest.approx([0.4, 0.4 * 100 ** 0.7])


def test_lineup_value_fills_dedicated_then_flex():
    lg = _league()
    by_pos = {"QB": [20, 18], "RB": [15, 10, 9], "WR": [14, 13, 12], "TE": [8, 7]}
    # QB 20 + RB 15+10 + WR 14+13 + TE 8 + FLEX max(9, 12, 7) = 12
    assert lineup_value(by_pos, lg) == pytest.approx(92.0)


def test_position_caps_limit_qb_and_te_to_one_spare():
    caps = position_caps(_league())
    assert caps[POS_INDEX["QB"]] == 2 and caps[POS_INDEX["TE"]] == 2
    assert caps[POS_INDEX["RB"]] == 8


def test_zero_noise_room_drafts_in_adp_order():
    rows = [("RB", float(i + 1), np.nan) for i in range(8)]
    pool = _pool(rows, board_order=[])
    lg = _league(starters={"QB": 0, "RB": 2, "WR": 0, "TE": 0, "FLEX": 0}, roster_size=2)
    sim = DraftSim(lg, pool, noise_scale=0.0)
    state = sim.run(sim.new_state(), sim.room(np.random.default_rng(0)))
    order = [i for k in range(len(sim.order)) for i in [state.rosters[sim.order[k]][
        sum(1 for j in range(k) if sim.order[j] == sim.order[k])]]]
    assert order == list(range(8))


def test_zero_noise_gives_zero_pick_spread_and_calibration_finds_the_generating_scale():
    rows = [(("RB", "WR")[i % 2], float(i + 1), np.nan) for i in range(60)]
    lg = _league(starters={"QB": 0, "RB": 2, "WR": 2, "TE": 0, "FLEX": 0}, roster_size=10)
    sim = DraftSim(lg, _pool(rows, board_order=[]), noise_scale=0.0)
    mean, sd = simulated_pick_spread(sim, n_drafts=5, rng=np.random.default_rng(0))
    assert np.allclose(sd[:40], 0.0) and np.allclose(mean[:40], np.arange(1, 41))

    sim.noise_scale = 1.0
    _, observed = simulated_pick_spread(sim, n_drafts=200, rng=np.random.default_rng(7))
    best, scores = calibrate_noise(sim, observed, grid=(0.5, 1.0, 2.0), n_drafts=200, seed=11)
    assert best == 1.0 and scores[1.0] < scores[0.5] and scores[1.0] < scores[2.0]
    assert sim.noise_scale == 1.0   # calibration restores the scale it was given


def test_adp_policy_respects_caps():
    rows = [("QB", 1.0, 20.0), ("QB", 2.0, 19.0), ("QB", 3.0, 18.0), ("RB", 4.0, 10.0)]
    lg = _league(starters={"QB": 1, "RB": 1, "WR": 0, "TE": 0, "FLEX": 0}, roster_size=4)
    sim = DraftSim(lg, _pool(rows), noise_scale=0.0)
    state = sim.new_state()
    for i in (0, 1):
        sim.take(state, 0, i)
    assert AdpPolicy().pick(sim, state, 0, 2) == 3   # a third QB is over the cap


def test_board_policy_skips_players_who_add_nothing_to_the_lineup():
    rows = [("RB", 9.0, 30.0), ("RB", 8.0, 25.0), ("WR", 7.0, 5.0)]
    lg = _league(starters={"QB": 0, "RB": 1, "WR": 1, "TE": 0, "FLEX": 0}, roster_size=3)
    sim = DraftSim(lg, _pool(rows), noise_scale=0.0)
    state = sim.new_state()
    sim.take(state, 0, 0)
    assert BoardPolicy().pick(sim, state, 0, 1) == 2   # second RB can't start; the WR can


def test_lookahead_waits_on_the_player_the_market_will_let_fall():
    """The market wants WRs; the model loves an RB nobody else will take.

    4 teams, 2 rounds, lineup RB + WR. Board: take RB_a (10) first, then the only WR left is
    WR_g (3) → 13. Lookahead: take WR_a (9) first, RB_a is still there at pick 8 → 19.
    """
    wrs = [("WR", float(i + 1), 9.0 - i) for i in range(7)]            # adp 1..7, value 9..3
    rbs = [("RB", 20.0 + i, 10.0 - 0.1 * i) for i in range(4)]         # adp 20.., value 10, 9.9..
    pool = _pool(wrs + rbs)
    lg = _league(starters={"QB": 0, "RB": 1, "WR": 1, "TE": 0, "FLEX": 0}, roster_size=2)
    sim = DraftSim(lg, pool, noise_scale=0.0)
    read = sim.room(np.random.default_rng(0))

    board = sim.run(sim.new_state(), read, me=0, policy=BoardPolicy())
    look = sim.run(sim.new_state(), read, me=0,
                   policy=LookaheadPolicy(n_plan_draws=1, rng=np.random.default_rng(1)))
    assert sim.my_value(board, 0) == pytest.approx(13.0)
    assert sim.my_value(look, 0) == pytest.approx(19.0)


def test_unscored_players_are_never_drafted_by_model_policies():
    rows = [("RB", 1.0, np.nan), ("RB", 2.0, 5.0), ("WR", 3.0, 4.0)]
    lg = _league(starters={"QB": 0, "RB": 1, "WR": 1, "TE": 0, "FLEX": 0}, roster_size=2)
    sim = DraftSim(lg, _pool(rows), noise_scale=0.0)
    state = sim.new_state()
    assert BoardPolicy().pick(sim, state, 0, 0) == 1
    assert 0 not in LookaheadPolicy(n_plan_draws=1).candidates(sim, state, 0)
    assert AdpPolicy().pick(sim, state, 0, 0) == 0    # the market drafter takes the rookie


def test_adp_policy_fills_an_open_starting_slot_before_best_adp():
    rows = [("RB", 1.0, 10.0), ("RB", 2.0, 9.0), ("TE", 50.0, 5.0)]
    lg = _league(starters={"QB": 0, "RB": 1, "WR": 0, "TE": 1, "FLEX": 0}, roster_size=3)
    sim = DraftSim(lg, _pool(rows), noise_scale=0.0)
    state = sim.new_state()
    sim.take(state, 0, 0)
    assert AdpPolicy().pick(sim, state, 0, 1) == 2    # the TE slot is open; a second RB can't start
    sim.take(state, 0, 2)
    assert AdpPolicy().pick(sim, state, 0, 2) == 1    # lineup full -> best ADP again


def test_insurance_bench_covers_the_absence_that_hurts_most():
    """QB + RB lineup already filled (QB 20, RB 10). Neither backup starts. The board bench takes
    the next board name (backup RB, 9); insurance takes the backup QB (15), because losing the QB
    costs 20 and losing the RB costs 10: mean gain QB (15+0)/2 = 7.5 > RB (0+9)/2 = 4.5."""
    rows = [("QB", 1.0, 20.0), ("RB", 2.0, 10.0), ("RB", 3.0, 9.0), ("QB", 4.0, 15.0)]
    lg = _league(starters={"QB": 1, "RB": 1, "WR": 0, "TE": 0, "FLEX": 0}, roster_size=4)
    sim = DraftSim(lg, _pool(rows, board_order=[0, 1, 2, 3]), noise_scale=0.0)
    state = sim.new_state()
    sim.take(state, 0, 0)
    sim.take(state, 0, 1)
    assert BoardPolicy().pick(sim, state, 0, 2) == 2
    assert BoardPolicy(bench="insurance").pick(sim, state, 0, 2) == 3
    with pytest.raises(ValueError, match="bench"):
        BoardPolicy(bench="vibes")


def test_market_window_takes_the_models_best_player_inside_the_markets_round():
    from position_predictor.eval.draftsim import MarketWindowPolicy

    # 4 teams -> a one-round window of 4 market ranks past the ADP pick (RB at adp 1).
    rows = [("RB", 1.0, 5.0),      # the market's pick
            ("WR", 2.0, 30.0),     # another position: never considered
            ("RB", 3.0, 9.0),      # inside the window -> the model's choice
            ("RB", 9.0, 20.0)]     # the model's favourite, but a reach past the window
    lg = _league(starters={"QB": 0, "RB": 1, "WR": 1, "TE": 0, "FLEX": 0}, roster_size=2)
    sim = DraftSim(lg, _pool(rows), noise_scale=0.0)
    assert MarketWindowPolicy().pick(sim, sim.new_state(), 0, 0) == 2
    assert MarketWindowPolicy(window=10).pick(sim, sim.new_state(), 0, 0) == 3


def test_market_window_falls_back_to_the_market_when_the_model_scores_no_one_in_range():
    from position_predictor.eval.draftsim import MarketWindowPolicy

    rows = [("RB", 1.0, np.nan),   # a rookie the model can't score
            ("RB", 20.0, 15.0)]    # scored, but far outside the window
    lg = _league(starters={"QB": 0, "RB": 1, "WR": 0, "TE": 0, "FLEX": 0}, roster_size=1)
    sim = DraftSim(lg, _pool(rows), noise_scale=0.0)
    assert MarketWindowPolicy().pick(sim, sim.new_state(), 0, 0) == 0
