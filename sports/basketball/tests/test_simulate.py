"""Tests for the Phase-2 draft simulator + representation shootout (pure helpers + planted signal)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nba_archetypes.eval.phase2_model import representation_shootout  # noqa: E402
from nba_archetypes.eval.simulate import (  # noqa: E402
    ZCOLS, attach_prior, cat_win_rates, player_value_table, simulate_draft, strategy_leaderboard,
    strategy_weights, team_coverage, team_outcomes)


def _player_seasons(n_per_season=40, seasons=(2014, 2015, 2016), seed=0):
    """Synthetic eligible player-seasons with the columns ``player_value_table`` consumes.

    Player ``athlete_id`` persists across seasons so ``attach_prior`` finds returning players.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for s in seasons:
        for pid in range(n_per_season):
            rows.append({
                "athlete_id": pid, "season": s, "eligible": True, "gp": 70,
                "pts_pg": rng.uniform(5, 30), "reb_pg": rng.uniform(2, 12),
                "ast_pg": rng.uniform(1, 9), "stl_pg": rng.uniform(0.3, 2.0),
                "blk_pg": rng.uniform(0.1, 2.5), "fg3m_pg": rng.uniform(0, 4),
                "tov_pg": rng.uniform(0.5, 4), "fgm_pg": rng.uniform(2, 10),
                "fga_pg": rng.uniform(5, 20), "ftm_pg": rng.uniform(1, 6),
                "fta_pg": rng.uniform(1, 8), "fg_pct": rng.uniform(0.4, 0.6),
                "ft_pct": rng.uniform(0.6, 0.9)})
    return pd.DataFrame(rows)


def test_player_value_table_zscores_center_per_season():
    vt = player_value_table(_player_seasons())
    for _, g in vt.groupby("season"):
        for c in ZCOLS:
            assert abs(g[c].mean()) < 1e-9          # standardized within season
        assert np.allclose(g["value"], g[ZCOLS].sum(axis=1))


def test_attach_prior_keeps_returning_players_only():
    vt = player_value_table(_player_seasons(seasons=(2014, 2015)))
    vp = attach_prior(vt)
    assert set(vp["season"].unique()) == {2015}     # 2014 has no prior -> dropped by inner join
    assert all(f"prior_{c}" in vp.columns for c in ZCOLS)
    assert len(vp) == 40                            # all 40 players returned from 2014


def test_strategy_weights_zeroes_punted_categories():
    w = strategy_weights("punt_ft")
    assert w[ZCOLS.index("z_ft")] == 0.0
    assert w.sum() == len(ZCOLS) - 1                # exactly one category punted


def test_cat_win_rates_average_half_and_symmetric():
    rng = np.random.default_rng(1)
    vecs = rng.normal(size=(12, 9))
    rates = cat_win_rates(vecs)
    assert abs(rates.mean() - 0.5) < 1e-9           # zero-sum: league mean is 0.5 by construction
    assert (rates >= 0).all() and (rates <= 1).all()


def test_team_coverage_sums_roster_z_actual_and_prior():
    vt = player_value_table(_player_seasons(seasons=(2014, 2015)))
    pool = attach_prior(vt)
    picks = {0: pool["athlete_id"].iloc[:5].tolist(), 1: pool["athlete_id"].iloc[5:10].tolist()}
    actual, prior = team_coverage(picks, pool, 2)
    assert actual.shape == (2, 9) and prior.shape == (2, 9)
    by_id = pool.set_index("athlete_id")
    expected = by_id.loc[picks[0]][ZCOLS].to_numpy().sum(axis=0)
    assert np.allclose(actual[0], expected)


def test_team_outcomes_negates_tov_and_computes_percentages():
    vt = player_value_table(_player_seasons(seasons=(2014, 2015)))
    pool = attach_prior(vt)
    picks = {0: pool["athlete_id"].iloc[:6].tolist()}
    vecs = team_outcomes(picks, pool, 1)
    assert vecs.shape == (1, 9)
    assert vecs[0, 8] < 0                           # TOV is negated (higher = better orientation)
    assert 0 <= vecs[0, 6] <= 1 and 0 <= vecs[0, 7] <= 1   # FT% and FG% are ratios


def test_simulate_draft_no_duplicate_picks():
    vt = player_value_table(_player_seasons(seasons=(2014, 2015)))
    pool = attach_prior(vt)
    rng = np.random.default_rng(3)
    picks = simulate_draft(pool, n_teams=4, n_rounds=5, team_strats=["balanced"] * 4,
                           team_temps=[0.1] * 4, rng=rng)
    allp = [p for ps in picks.values() for p in ps]
    assert len(allp) == len(set(allp)) == 20        # 4*5 distinct players, no double-draft


def test_strategy_leaderboard_sorted_and_z_signed():
    labels = pd.DataFrame({
        "strategy": ["A"] * 50 + ["B"] * 50,
        "sim_cat_win_rate": [0.6] * 50 + [0.4] * 50})
    board = strategy_leaderboard(labels)
    assert [r["strategy"] for r in board] == ["A", "B"]   # higher mean first
    assert board[0]["z_vs_0.5"] > 0 and board[1]["z_vs_0.5"] < 0


def test_representation_shootout_recovers_coverage_signal():
    # Target driven by actual coverage; archetype shares are noise -> shootout must separate them.
    rng = np.random.default_rng(7)
    rows = []
    for s in range(6):                              # 6 seasons (groups) x 30 teams
        for t in range(30):
            cov_act = rng.normal(size=9)
            y = 0.5 + 0.05 * cov_act[0] + rng.normal(0, 0.005)
            rec = {"season": s, "sim_cat_win_rate": y}
            for j, c in enumerate(ZCOLS):
                rec[f"cov_act_{c[2:]}"] = cov_act[j]
                rec[f"cov_pri_{c[2:]}"] = cov_act[j] + rng.normal(0, 0.5)   # noisy prior
            for k in range(12):
                rec[f"comp_arch{k}"] = rng.random()
            rows.append(rec)
    res = representation_shootout(pd.DataFrame(rows))
    assert res["actual coverage"]["oof_r2"] > 0.5         # real coverage signal recovered
    assert res["actual coverage"]["oof_r2"] > res["archetype shares"]["oof_r2"]
    assert res["archetype shares"]["oof_r2"] < 0.1        # shares are noise here
