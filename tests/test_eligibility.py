"""Tests for Stage 5 eligibility-cutoff logic (pure; deterministic via seed)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from position_predictor.eligibility.cutoff import (  # noqa: E402
    choose_games_cutoff,
    coverage_purity,
    snap_coverage,
    split_half_reliability,
)


def _weekly_with_signal(noise_sd, n_players=120, games=16, seed=0):
    """Synthetic weekly PPR: each player has a stable true mean + per-game noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_players):
        true_mean = rng.normal(12, 5)
        for w in range(1, games + 1):
            rows.append(dict(player_id=f"P{p}", season=2020, week=w, season_type="REG",
                             fantasy_points_ppr=true_mean + rng.normal(0, noise_sd)))
    return pd.DataFrame(rows)


def test_reliability_increases_with_k():
    rel = split_half_reliability(_weekly_with_signal(noise_sd=6.0), [2, 4, 8],
                                 n_repeats=40, seed=1)
    by_k = rel.set_index("k")["reliability"]
    # more games -> more stable estimate -> higher reliability
    assert by_k[2] < by_k[8]
    assert (by_k > 0).all()


def test_reliability_high_for_low_noise_low_for_pure_noise():
    # almost no per-game noise -> a few games already reliable
    clean = split_half_reliability(_weekly_with_signal(noise_sd=0.5), [4], n_repeats=40, seed=2)
    assert clean.set_index("k").loc[4, "reliability"] > 0.9
    # pure noise (no between-player signal) -> reliability ~ 0
    rng = np.random.default_rng(3)
    rows = [dict(player_id=f"P{p}", season=2020, week=w, season_type="REG",
                 fantasy_points_ppr=rng.normal(10, 5))
            for p in range(120) for w in range(1, 17)]
    noise = split_half_reliability(pd.DataFrame(rows), [4], n_repeats=40, seed=3)
    assert abs(noise.set_index("k").loc[4, "reliability"]) < 0.2


def test_split_half_excludes_playoffs_and_small_pool():
    wk = _weekly_with_signal(noise_sd=4.0, n_players=6, games=4)  # only 4 games each
    # k=4 needs 8 games -> pool empty -> NaN reliability, n_pool 0
    rel = split_half_reliability(wk, [4], n_repeats=10, seed=1)
    assert rel.iloc[0]["n_pool"] == 0
    assert pd.isna(rel.iloc[0]["reliability"])


def test_choose_games_cutoff_threshold_and_fallback():
    rel = pd.DataFrame({"k": [2, 4, 6, 8],
                        "reliability": [0.4, 0.6, 0.72, 0.8], "n_pool": [100] * 4})
    g, reached = choose_games_cutoff(rel, target=0.70)
    assert g == 6 and reached is True
    # nothing clears a high target -> fall back to the most reliable k, reached False
    g2, reached2 = choose_games_cutoff(rel, target=0.99)
    assert g2 == 8 and reached2 is False


def test_coverage_purity_math():
    season = pd.DataFrame([
        dict(player_id="A", games=16, ppr_points=300.0, ppg=18.8),
        dict(player_id="B", games=10, ppr_points=150.0, ppg=15.0),
        dict(player_id="C", games=3, ppr_points=60.0, ppg=20.0),   # high ppg, few games
        dict(player_id="D", games=2, ppr_points=10.0, ppg=5.0),
    ])
    cov = coverage_purity(season, [4, 12]).set_index("games_cutoff")
    # G>=4 keeps A,B (450/520 of points); excludes C,D
    assert cov.loc[4, "n_eligible"] == 2
    assert abs(cov.loc[4, "coverage_points"] - 450 / 520) < 1e-9
    # C (ppg 20) is excluded but >= the included median ppg -> counts as excluded_relevant
    assert cov.loc[4, "excluded_relevant"] == 1


def test_snap_coverage_grid():
    season = pd.DataFrame([
        dict(snap_share=0.8, ppr_points=300.0),
        dict(snap_share=0.45, ppr_points=150.0),
        dict(snap_share=0.2, ppr_points=50.0),
        dict(snap_share=np.nan, ppr_points=40.0),  # pre-2012 / unmapped -> dropped
    ])
    cov = snap_coverage(season, [0.30, 0.50]).set_index("snap_cutoff")
    assert cov.loc[0.30, "n_eligible"] == 2   # 0.8 and 0.45 clear 0.30
    assert cov.loc[0.50, "n_eligible"] == 1   # only 0.8 clears 0.50
