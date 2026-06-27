"""Tests for the handcuff-selection tool (eval/handcuff.py)."""
import numpy as np
import pandas as pd

from position_predictor.eval.handcuff import (
    GAMES_TARGET,
    SIGNALS,
    _predict_durability,
    _predict_prior_games,
    _season_length,
    backtest_risk_signals,
    build_handcuff_board,
    build_injury_risk_list,
    project_risk,
)


def test_season_length_17_from_2021():
    assert _season_length(2020) == 16
    assert _season_length(2021) == 17
    assert _season_length(2025) == 17


def test_deterministic_signals():
    score = pd.DataFrame({
        "season": [2024, 2019],
        "games": [12.0, 8.0],
        "avail_rate_3yr": [0.5, 1.0],
        "availability_rate": [0.4, 0.9],
    })
    # prior_games: next ≈ this season's games.
    assert list(_predict_prior_games(None, score)) == [12.0, 8.0]
    # durability: 3yr availability rate × NEXT season's length (2025→17, 2020→16).
    np.testing.assert_allclose(_predict_durability(None, score, horizon=1), [17 * 0.5, 16 * 1.0])


def test_project_risk_games_missed_and_share():
    board = pd.DataFrame({"player_id": ["a", "b"], "season": [2025, 2025],
                          "games": [16.0, 4.0], "avail_rate_3yr": [1.0, 0.3],
                          "availability_rate": [1.0, 0.3], GAMES_TARGET: [np.nan, np.nan]})
    risk = project_risk(board, [], board, board_season=2025, signal="prior_games",
                        horizon=1, seed=1)
    # 2026 season length = 17; prior_games predicts 16 / 4 → missed 1 / 13.
    a = risk.set_index("player_id").loc["a"]
    b = risk.set_index("player_id").loc["b"]
    assert a["exp_games_missed"] == 1.0 and b["exp_games_missed"] == 13.0
    assert b["miss_share"] == round(13 / 17, 3)  # 0.765


def test_project_risk_clips_to_season_length():
    board = pd.DataFrame({"player_id": ["a"], "season": [2025], "games": [25.0],
                          "avail_rate_3yr": [1.0], "availability_rate": [1.0],
                          GAMES_TARGET: [np.nan]})
    risk = project_risk(board, [], board, board_season=2025, signal="prior_games",
                        horizon=1, seed=1)
    assert risk.iloc[0]["pred_games_next"] == 17.0  # clipped, never negative missed
    assert risk.iloc[0]["exp_games_missed"] == 0.0


def _board_inputs():
    rb_proj = pd.DataFrame({
        "player_id": ["A", "B", "C", "D", "E", "F", "G"],
        "player_name": ["Star A", "Cuff B", "Star C", "Cuff D", "Solo E", "Deep F", "Cuff G"],
        "position": ["RB"] * 7,
        "proj_ppg": [18.0, 6.0, 10.0, 5.0, 4.0, 3.0, 2.0],
        "proj_pos_rank": [2, 40, 10, 50, 60, 90, 95],
    })
    features_board = pd.DataFrame({
        "player_id": ["A", "B", "C", "D", "E", "F", "G"],
        "recent_team": ["AAA", "AAA", "BBB", "BBB", "CCC", "DDD", "DDD"],
    })
    risk = pd.DataFrame({
        "player_id": ["A", "B", "C", "D", "E", "F", "G"],
        "pred_games_next": [11.9, 0, 15.3, 0, 0, 0, 0],
        "exp_games_missed": [5.1, 0, 1.7, 0, 0, 0, 0],
        "miss_share": [0.3, 0.0, 0.1, 0.0, 0.0, 0.5, 0.0],
    })
    return rb_proj, features_board, risk


def test_build_board_starter_backup_and_contingent_upside():
    rb_proj, features_board, risk = _board_inputs()
    board = build_handcuff_board(rb_proj, features_board, risk, max_starter_rank=36)
    # CCC has only 1 RB → dropped; DDD's starter (rank 90) > 36 → dropped. AAA + BBB remain.
    assert list(board["team"]) == ["AAA", "BBB"]  # sorted by contingent upside
    top = board.iloc[0]
    assert top["starter"] == "Star A" and top["handcuff"] == "Cuff B"
    # contingent = (18 - 6) * 0.3 = 3.6 ; handcuff_value = 6 + 3.6 = 9.6
    assert top["contingent_upside"] == 3.6 and top["handcuff_value"] == 9.6
    assert list(board["rank"]) == [1, 2]
    # BBB: (10 - 5) * 0.1 = 0.5
    assert board.iloc[1]["contingent_upside"] == 0.5


def test_build_board_empty_when_no_eligible_starters():
    rb_proj, features_board, risk = _board_inputs()
    board = build_handcuff_board(rb_proj, features_board, risk, max_starter_rank=1)
    assert board.empty


def test_backtest_returns_all_signals_and_valid_winner():
    rng = np.random.default_rng(0)
    rows = []
    for season in range(2016, 2025):
        for i in range(40):
            games = float(rng.integers(1, 18))
            rows.append({
                "season": season, "player_id": f"{season}_{i}",
                "games": games, "avail_rate_3yr": games / 17, "availability_rate": games / 17,
                "x1": rng.normal(), GAMES_TARGET: float(rng.integers(1, 18)),
            })
    df = pd.DataFrame(rows)
    table, winner = backtest_risk_signals(df, ["games", "avail_rate_3yr", "x1"],
                                          cutoff=4, horizon=1, seed=1, min_test=5)
    assert set(table["signal"]) == set(SIGNALS)
    assert winner in SIGNALS
    # table sorted by clears_auc desc → the winner has the max AUC.
    assert table.iloc[0]["signal"] == winner


def test_injury_risk_list_ranks_tiers_and_filters_starters():
    # 8 projected starters + 1 non-starter (rank 40, must be dropped by top_starters=8).
    proj = pd.DataFrame({
        "player_id": list("abcdefgh") + ["z"],
        "player_name": [f"QB{c}" for c in "abcdefgh"] + ["Backup Z"],
        "position": ["QB"] * 9,
        "proj_ppg": [18, 17, 16, 15, 14, 13, 12, 11, 5.0],
        "proj_pos_rank": [1, 2, 3, 4, 5, 6, 7, 8, 40],
    })
    risk = pd.DataFrame({
        "player_id": list("abcdefgh") + ["z"],
        "pred_games_next": [5, 7, 9, 10, 11, 12, 13, 14, 2.0],
        "exp_games_missed": [12, 10, 8, 7, 6, 5, 4, 3, 15.0],
        "miss_share": [0.7, 0.6, 0.47, 0.41, 0.35, 0.29, 0.24, 0.18, 0.88],
    })
    rl = build_injury_risk_list(proj, risk, top_starters=8)
    assert len(rl) == 8                                   # non-starter z dropped
    assert "z" not in set(rl["player_name"])
    # ranked by exp_games_missed desc: most at-risk first
    assert list(rl["player_name"])[:2] == ["QBa", "QBb"]
    assert list(rl["risk_rank"]) == list(range(1, 9))
    # quartile tiers: High = top 25% (2 of 8), Lower = bottom 25% (2 of 8)
    assert list(rl["risk_tier"]) == ["High", "High", "Moderate", "Moderate",
                                     "Moderate", "Moderate", "Lower", "Lower"]
    # draft_backup flags exactly the High tier
    assert list(rl["draft_backup"]) == [True, True, False, False, False, False, False, False]


def test_injury_risk_list_empty_when_no_starters():
    proj = pd.DataFrame({"player_id": ["a"], "player_name": ["QBa"], "position": ["QB"],
                         "proj_ppg": [10.0], "proj_pos_rank": [50]})
    risk = pd.DataFrame({"player_id": ["a"], "pred_games_next": [8.0],
                         "exp_games_missed": [9.0], "miss_share": [0.5]})
    assert build_injury_risk_list(proj, risk, top_starters=32).empty
