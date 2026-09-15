"""Tests for the draft backtest's pure pieces (synthetic boards/rosters/points; no network/model)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from position_predictor.eval.draft_backtest import (  # noqa: E402
    backtest_league,
    build_pool,
    match_board_ids,
    summarize,
    team_points_per_week,
    weekly_matrix,
)
from position_predictor.eval.draftsim import POS_INDEX  # noqa: E402


def _roster_row(team, position, full, football, last, pid, season=2023):
    return dict(season=season, team=team, position=position, full_name=full,
                football_name=football, last_name=last, player_id=pid)


def _rosters():
    return pd.DataFrame([
        _roster_row("CIN", "WR", "Ja'Marr Chase", "Ja'Marr", "Chase", "C1"),
        _roster_row("NO", "QB", "Taysom Hill", "Taysom", "Hill", "H1"),
        _roster_row("ARI", "WR", "Marquise Brown", "Marquise", "Brown", "B1"),
        _roster_row("PHI", "WR", "A.J. Brown", "A.J.", "Brown", "AJ1"),
        _roster_row("JAX", "WR", "Christian Kirk", "Christian", "Kirk", "K1"),
        _roster_row("DAL", "RB", "Tony Pollard", "Tony", "Pollard", "P1"),
        _roster_row("DAL", "RB", "Rico Pollard", "Rico", "Pollard", "P2"),
    ])


def _board(rows):
    """rows: (name, position, team, adp)."""
    return pd.DataFrame([dict(ffc_id=i, name=n, position=p, team=t, adp=a, stdev=1.0, season=2023)
                         for i, (n, p, t, a) in enumerate(rows)])


def test_match_board_ids_three_passes():
    board = _board([("Ja'Marr Chase", "WR", "CIN", 1.0),      # 1: exact name + position
                    ("Taysom Hill", "TE", "NO", 150.0),        # 2: FFC calls him a TE
                    ("Hollywood Brown", "WR", "ARI", 60.0),    # 3: nickname, team + surname
                    ("Chris Kirk", "WR", "JAC", 90.0),         # 3: via the JAC -> JAX alias
                    ("Kenny Pollard", "RB", "DAL", 99.0),      # two Pollards on DAL -> none
                    ("Nobody Special", "RB", "SEA", 170.0)])
    out = match_board_ids(board, _rosters()).set_index("name")
    assert out.loc["Ja'Marr Chase", ["player_id", "match_pass"]].tolist() == ["C1", 1]
    assert out.loc["Taysom Hill", ["player_id", "match_pass"]].tolist() == ["H1", 2]
    assert out.loc["Hollywood Brown", ["player_id", "match_pass"]].tolist() == ["B1", 3]
    assert out.loc["Chris Kirk", ["player_id", "match_pass"]].tolist() == ["K1", 3]
    for unmatched in ("Kenny Pollard", "Nobody Special"):
        assert out.loc[unmatched, "player_id"] is None and out.loc[unmatched, "match_pass"] == 0


def test_build_pool_merges_market_and_model_boards():
    proj = pd.DataFrame([
        dict(player_id="R1", player_name="Rb One", position="RB", proj_ppg=20.0, proj_pos_rank=1),
        dict(player_id="W2", player_name="Wr Two", position="WR", proj_ppg=18.0, proj_pos_rank=1),
        dict(player_id="R3", player_name="Rb Three", position="RB", proj_ppg=12.0, proj_pos_rank=2),
        dict(player_id="Q4", player_name="Qb Four", position="QB", proj_ppg=22.0, proj_pos_rank=1),
    ])
    board = pd.DataFrame([
        dict(ffc_id=1, name="Rb One", position="RB", adp=1.0, stdev=0.5, player_id="R1"),
        dict(ffc_id=2, name="Wr Two", position="WR", adp=2.5, stdev=0.8, player_id="W2"),
        dict(ffc_id=3, name="Rookie Guy", position="RB", adp=3.0, stdev=1.0, player_id=None),
        dict(ffc_id=4, name="Some Kicker", position="PK", adp=4.0, stdev=2.0, player_id=None),
        dict(ffc_id=5, name="Rb Three", position="RB", adp=6.0, stdev=1.5, player_id="R3"),
    ])
    frame, pool = build_pool(board, proj, backtest_league(10, "ppr"))
    by = frame.set_index("player_id")
    # K/DST leave the pool and the market ranks close up behind them.
    assert "Some Kicker" not in set(frame["name"])
    assert by["market_rank"].to_dict() == pytest.approx(
        {"R1": 1.0, "W2": 2.0, "ffc:3": 3.0, "R3": 4.0, "Q4": np.nan}, nan_ok=True)
    # The rookie is draftable by the room but unscored; the model-only QB is scored but unranked.
    assert np.isnan(by.at["ffc:3", "proj_ppg"]) and by.at["Q4", "proj_ppg"] == 22.0
    rookie = frame.index[frame["player_id"] == "ffc:3"][0]
    assert rookie not in pool.board_order and not pool.scored[rookie]
    assert len(pool.board_order) == 4 and frame.at[pool.board_order[0], "player_id"] == "R1"
    assert pool.position[frame.index[frame["player_id"] == "Q4"][0]] == POS_INDEX["QB"]


def test_weekly_matrix_places_points_and_zero_fills():
    pts = pd.DataFrame([dict(player_id="A", season=2023, week=1, points=10.0),
                        dict(player_id="A", season=2023, week=2, points=20.0),
                        dict(player_id="B", season=2023, week=2, points=5.0),
                        dict(player_id="A", season=2022, week=1, points=99.0)])
    m = weekly_matrix(np.array(["B", "A", "X"]), pts, 2023)
    assert m.tolist() == [[0.0, 5.0], [10.0, 20.0], [0.0, 0.0]]


def test_team_points_per_week_is_the_best_lineup_each_week():
    # QB, RB a/b/c, WR x/y, TE; RB a is on bye in week 2.
    positions = np.array([POS_INDEX[p] for p in ["QB", "RB", "RB", "RB", "WR", "WR", "TE"]])
    points = np.array([[20, 10], [15, 0], [5, 12], [8, 9], [10, 10], [6, 7], [4, 3]], float)
    # wk1: 20 + (15+8) + (10+6) + 4 + flex 5 = 68 ; wk2: 10 + (12+9) + (10+7) + 3 + flex 0 = 51
    got = team_points_per_week(range(7), points, positions, backtest_league(10, "ppr"))
    assert got == pytest.approx((68 + 51) / 2)


def test_weekly_matrix_played_marks_zero_point_games():
    pts = pd.DataFrame([dict(player_id="A", season=2023, week=1, points=0.0),
                        dict(player_id="A", season=2023, week=2, points=7.0),
                        dict(player_id="B", season=2023, week=2, points=3.0)])
    m, played = weekly_matrix(np.array(["A", "B"]), pts, 2023, with_played=True)
    assert m.tolist() == [[0.0, 7.0], [0.0, 3.0]]
    assert played.tolist() == [[True, True], [False, True]]   # A's 0-point game still counts


def test_managed_lineup_starts_by_season_ppg_not_by_the_week():
    """QB2 plays week 1 only (ppg 25). RB2 averages 11 but scores 2 in week 1, when RB3 (ppg 5)
    scores 9 — a managed lineup starts RB2 anyway; best ball takes RB3's spike.

    managed:  wk1 25 + (2+15) + (10+7) + 4 + flex WR2 6 = 69 ; wk2 30 + (20+5) + 17 + 4 + 6 = 82
    bestball: wk1 25 + (15+9) + 17 + 4 + 6 = 76              ; wk2 82
    """
    positions = np.array([POS_INDEX[p] for p in
                          ["QB", "QB", "RB", "RB", "RB", "WR", "WR", "WR", "TE"]])
    points = np.array([[10, 30], [25, 0], [15, 5], [2, 20], [9, 1],
                       [10, 10], [6, 6], [7, 7], [4, 4]], float)
    played = np.ones(points.shape, dtype=bool)
    played[1, 1] = False
    lg = backtest_league(10, "ppr")
    managed = team_points_per_week(range(9), points, positions, lg, played=played,
                                   lineup="managed")
    bestball = team_points_per_week(range(9), points, positions, lg, lineup="bestball")
    assert managed == pytest.approx((69 + 82) / 2)
    assert bestball == pytest.approx((76 + 82) / 2)
    with pytest.raises(ValueError, match="played"):
        team_points_per_week(range(9), points, positions, lg, lineup="managed")
    with pytest.raises(ValueError, match="lineup"):
        team_points_per_week(range(9), points, positions, lg, lineup="hunch")


def test_summarize_pairs_policies_and_clusters_by_season():
    rows = []
    for season, pts in [(2022, {"adp": 100, "vorp_board": 101, "lookahead": 103}),
                        (2023, {"adp": 100, "vorp_board": 99, "lookahead": 102})]:
        for policy, p in pts.items():
            rows.append(dict(scoring="ppr", teams=12, season=season, draw=0, slot=1,
                             policy=policy, points_pw=p, league_mean_pw=100.0, rank=1,
                             proj_lineup=90.0))
    gains = summarize(pd.DataFrame(rows))["gains"].set_index("comparison")
    look = gains.loc["lookahead − vorp_board"]
    assert look["mean"] == pytest.approx(2.5) and look["se"] == pytest.approx(0.5)
    assert look["seasons_won"] == 2 and look["seasons"] == 2
    board = gains.loc["vorp_board − adp"]
    assert board["mean"] == pytest.approx(0.0) and board["seasons_won"] == 1


def test_ranked_only_pool_makes_market_unranked_players_undraftable_for_me():
    from position_predictor.eval.draft_backtest import make_policy, ranked_only_pool
    from position_predictor.eval.draftsim import Pool

    pool = Pool(position=[0, 1, 2], adp=[1.0, np.nan, 2.0], value=[20.0, 15.0, np.nan],
                board_order=[0, 1], player_id=np.array(["a", "b", "c"]))
    ranked = ranked_only_pool(pool)
    assert ranked.board_order.tolist() == [0]
    assert ranked.scored.tolist() == [True, False, False]
    assert np.array_equal(ranked._adp_filled, pool._adp_filled)   # the room reads the same board
    assert make_policy("vorp_board_ranked", None, 0, 0)[1] == "ranked"
    assert make_policy("adp", None, 0, 0)[1] == "all"
    with pytest.raises(ValueError, match="unknown policy"):
        make_policy("coinflip", None, 0, 0)


def test_render_markdown_reports_gains_policies_and_seasons():
    from position_predictor.eval.draft_backtest import render_markdown

    rows = []
    for season in (2022, 2023):
        for policy, pts, rank in [("adp", 100.0, 5), ("vorp_board_ranked", 102.0, 3)]:
            rows.append(dict(scoring="half_ppr", teams=10, season=season, draw=0, slot=1,
                             policy=policy, points_pw=pts, league_mean_pw=100.0, rank=rank,
                             proj_lineup=95.0))
    infos = [dict(scoring="half_ppr", season=2022, teams=10, ranked=117, picks=140, pool=180,
                  scored_share_top_picks=0.9, skipped="board ranks 117 players for 140 picks"),
             dict(scoring="half_ppr", season=2023, teams=10, ranked=175, picks=140, pool=260,
                  scored_share_top_picks=0.9, noise_scale=1.25)]
    md = render_markdown(summarize(pd.DataFrame(rows)), infos, draws=1)
    assert md.startswith("# Draft backtest")
    assert "| half_ppr | 10 | vorp_board_ranked − adp | 2.00 |" in md
    assert "2/2" in md                                         # seasons won
    assert "VORP board, market-ranked players only" in md      # policy legend
    assert "board ranks 117 players for 140 picks" in md       # skipped seasons are visible
