"""Tests for the Phase-2 roster optimizer (pure helpers + a planted-skill in-sim lift)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nba_archetypes.eval.optimize import (  # noqa: E402
    CATS, _contested_mask, _punt_cats, coverage_picker, evaluate_in_sim, expected_category_wins,
    field_coverage_stats, optimize_roster)
from nba_archetypes.eval.simulate import attach_prior, player_value_table  # noqa: E402


def _players_with_persistent_skill(n=60, seasons=(2014, 2015, 2016), seed=0):
    """Synthetic players whose stats track a latent per-player skill across seasons.

    Persistence makes the **prior** z-profile a real (noisy) signal for the **actual** one, so an
    optimizer drafting on prior coverage should out-produce a field that partly drafts at random.
    """
    rng = np.random.default_rng(seed)
    skill = {pid: rng.normal(size=9) for pid in range(n)}
    base = dict(pts=15, reb=6, ast=4, stl=1, blk=0.8, fg3m=1.5, tov=2, fgm=6, fga=13, ftm=3, fta=4)
    rows = []
    for s in seasons:
        for pid in range(n):
            sk = skill[pid] + rng.normal(0, 0.4, size=9)          # season noise around latent skill
            rows.append({
                "athlete_id": pid, "season": s, "eligible": True, "gp": 70,
                "player_name": f"P{pid}",
                "pts_pg": base["pts"] + 4 * sk[0], "reb_pg": max(0.5, base["reb"] + 2 * sk[1]),
                "ast_pg": max(0.2, base["ast"] + 1.5 * sk[2]), "stl_pg": max(0.1, base["stl"] + 0.4 * sk[3]),
                "blk_pg": max(0.05, base["blk"] + 0.4 * sk[4]), "fg3m_pg": max(0.0, base["fg3m"] + 1.0 * sk[5]),
                "tov_pg": max(0.3, base["tov"] + 0.6 * sk[6]),
                "fgm_pg": base["fgm"], "fga_pg": base["fga"], "ftm_pg": base["ftm"], "fta_pg": base["fta"],
                "fg_pct": 0.46 + 0.03 * sk[7], "ft_pct": 0.75 + 0.05 * sk[8]})
    return pd.DataFrame(rows)


def _field_table():
    """Small synthetic sim table with cov_act_* columns for field_coverage_stats."""
    rng = np.random.default_rng(1)
    data = {f"cov_act_{c}": rng.normal(2 * j, 1.0, size=200) for j, c in enumerate(CATS)}
    return pd.DataFrame(data)


def test_punt_cats_and_contested_mask():
    assert _punt_cats("punt_ft") == {"ft"}
    assert _punt_cats(("ft", "fg3m")) == {"ft", "fg3m"}
    assert _punt_cats(()) == set()
    mask = _contested_mask("punt_ft")
    assert mask[CATS.index("ft")] == False        # noqa: E712 — ft is conceded
    assert mask.sum() == len(CATS) - 1


def test_field_coverage_stats_shapes_and_values():
    mu, sigma = field_coverage_stats(_field_table(), "cov_act")
    assert mu.shape == (9,) and sigma.shape == (9,)
    assert mu[CATS.index("reb")] > mu[CATS.index("pts")]   # reb built with a larger mean above


def test_expected_category_wins_at_mean_is_half():
    mu = np.zeros(9)
    sigma = np.ones(9)
    per_cat, contested, overall = expected_category_wins(np.zeros(9), mu, sigma, punts="punt_ft")
    assert np.allclose(per_cat, 0.5)
    assert abs(contested - 0.5) < 1e-9 and abs(overall - 0.5) < 1e-9


def test_optimize_roster_distinct_respects_taken_and_punt():
    vp = attach_prior(player_value_table(_players_with_persistent_skill(seasons=(2014, 2015))))
    pool = vp[vp["season"] == 2015]
    mu, sigma = field_coverage_stats(_field_table(), "cov_act")
    taken = pool["athlete_id"].iloc[:5].tolist()
    r = optimize_roster(pool, mu, sigma, n_rounds=8, punts="punt_ft", taken=taken)
    assert len(r["athlete_ids"]) == 8 == len(set(r["athlete_ids"]))
    assert not set(r["athlete_ids"]) & set(taken)          # taken players excluded
    assert set(r["exp_win_by_cat"]) == set(CATS)


def test_coverage_picker_returns_available_indices():
    vp = attach_prior(player_value_table(_players_with_persistent_skill(seasons=(2014, 2015))))
    pool = vp[vp["season"] == 2015].reset_index(drop=True)
    mu, sigma = field_coverage_stats(_field_table(), "cov_act")
    pick = coverage_picker(pool, mu, sigma, punts=())
    available = np.ones(len(pool), dtype=bool)
    chosen = []
    for _ in range(6):
        idx = pick(available)
        assert available[idx]                              # never picks a taken slot
        available[idx] = False
        chosen.append(idx)
    assert len(set(chosen)) == 6


def test_evaluate_in_sim_optimizer_beats_field_with_persistent_skill():
    vp = attach_prior(player_value_table(_players_with_persistent_skill(seed=3)))
    mu, sigma = field_coverage_stats(_field_table(), "cov_act")
    _, _, summary = evaluate_in_sim(vp, mu, sigma, seasons=[2015, 2016], n_leagues=20, punts=(),
                                    n_teams=4, n_rounds=6, seed=5)
    assert summary["n_leagues"] > 0
    assert 0 <= summary["opt_mean"] <= 1 and 0 <= summary["field_mean"] <= 1
    assert summary["lift"] > 0                             # persistent skill -> prior draft wins
