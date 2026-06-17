"""Tests for Stage 6 EDA logic (pure functions; synthetic data with known signal)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from position_predictor.eda.analyze import (  # noqa: E402
    block_coverage,
    column_coverage_by_season,
    distribution_summary,
    ngs_coverage_by_tier,
    regression_to_mean,
    season_universe,
    target_stability,
)
from position_predictor.eras import Era  # noqa: E402

ERAS = [
    Era("boxscore", 1999, 2011, ("volume", "efficiency")),
    Era("snaps", 2012, 2015, ("volume", "efficiency", "snap_usage")),
    Era("ngs", 2016, None, ("volume", "efficiency", "snap_usage", "ngs_efficiency")),
]


def _season_df():
    """Two feature seasons: 2022 fully observed, 2023 fully censored (latest season)."""
    rows = []
    for i in range(10):
        rows.append(dict(player_id=f"P{i}", season=2022, games=12, ppg=10.0 + i,
                         status_next="active" if i < 7 else "retired",
                         target_ppg_next=(11.0 + i) if i < 7 else np.nan))
    for i in range(8):
        rows.append(dict(player_id=f"P{i}", season=2023, games=11, ppg=9.0 + i,
                         status_next="censored", target_ppg_next=np.nan))
    return pd.DataFrame(rows)


def test_season_universe_target_coverage_and_censoring():
    uni = season_universe(_season_df(), ERAS).set_index("season")
    # 2022: 7 of 10 active-next -> target coverage 0.7; era assigned from feature season
    assert uni.loc[2022, "n_player_seasons"] == 10
    assert uni.loc[2022, "n_active_next"] == 7
    assert abs(uni.loc[2022, "target_coverage"] - 0.7) < 1e-9
    assert uni.loc[2022, "era"] == "ngs"
    # 2023 is the latest season -> fully censored -> zero target coverage (not retirement)
    assert uni.loc[2023, "target_coverage"] == 0.0
    assert uni.loc[2023, "n_retired_next"] == 0


def test_block_coverage_era_gating():
    # snap_usage column populated only in the snaps/ngs eras; boxscore rows leave it NaN.
    df = pd.DataFrame([
        dict(season=2005, vol=1.0, snap=np.nan),
        dict(season=2014, vol=1.0, snap=0.5),
        dict(season=2018, vol=1.0, snap=0.6),
    ])
    block_columns = {"volume": ["vol"], "snap_usage": ["snap"]}
    cov = block_coverage(df, block_columns, ERAS).set_index(["era", "block"])
    assert cov.loc[("boxscore", "snap_usage"), "coverage"] == 0.0
    assert cov.loc[("boxscore", "snap_usage"), "in_era_schema"] is np.False_ or \
        cov.loc[("boxscore", "snap_usage"), "in_era_schema"] == False  # noqa: E712
    assert cov.loc[("snaps", "snap_usage"), "coverage"] == 1.0
    assert cov.loc[("ngs", "volume"), "coverage"] == 1.0


def test_column_coverage_by_season():
    df = pd.DataFrame([
        dict(season=2011, snap_share=np.nan),
        dict(season=2011, snap_share=np.nan),
        dict(season=2013, snap_share=0.4),
        dict(season=2013, snap_share=np.nan),
    ])
    cc = column_coverage_by_season(df, ["snap_share"]).set_index("season")
    assert cc.loc[2011, "coverage"] == 0.0
    assert cc.loc[2013, "coverage"] == 0.5


def test_ngs_coverage_by_tier():
    # Top-12 backs all have NGS rushing; fringe (37+) none.
    rows = []
    for r in range(1, 13):
        rows.append(dict(season=2020, finish_ppr_rank=r, has_ngs_rush=1, has_ngs_rec=1))
    for r in range(37, 57):
        rows.append(dict(season=2020, finish_ppr_rank=r, has_ngs_rush=0, has_ngs_rec=0))
    tier = ngs_coverage_by_tier(pd.DataFrame(rows)).set_index("tier")
    assert tier.loc["<= 12", "has_ngs_rush"] == 1.0
    assert tier.loc["37+", "has_ngs_rush"] == 0.0
    # pre-2016 rows are excluded from the NGS-era read
    pre = ngs_coverage_by_tier(pd.DataFrame([
        dict(season=2010, finish_ppr_rank=1, has_ngs_rush=1, has_ngs_rec=1)]))
    assert int(pre["n"].sum()) == 0


def test_distribution_summary_skew_and_zero():
    df = pd.DataFrame({"x": [0, 0, 0, 1, 2, 3, 50]})  # right-skewed with zero mass
    d = distribution_summary(df, ["x", "missing"]).set_index("column")
    assert "missing" not in d.index            # absent columns are skipped
    assert d.loc["x", "n"] == 7
    assert abs(d.loc["x", "pct_zero"] - 3 / 7) < 1e-9
    assert d.loc["x", "skew"] > 1.0            # long right tail


def test_target_stability_recovers_known_signal():
    # target = ppg + 1 (perfectly monotonic) -> Spearman 1.0, persistence MAE ~ 1.0
    df = pd.DataFrame({"season": [2020] * 6,
                       "ppg": [1, 2, 3, 4, 5, 6.0],
                       "target_ppg_next": [2, 3, 4, 5, 6, 7.0]})
    st = target_stability(df)
    row = st[st["season"] == 2020].iloc[0]
    assert abs(row["spearman"] - 1.0) < 1e-9
    assert abs(row["persistence_mae"] - 1.0) < 1e-9
    pooled = st[st["season"] == -1].iloc[0]   # pooled row present
    assert int(pooled["n_pairs"]) == 6


def test_regression_to_mean_reverts_tails():
    rng = np.random.default_rng(0)
    n = 500
    ppg = rng.normal(10, 5, n)
    # next season = half-shrink toward the mean (10) + noise
    target = 10 + 0.5 * (ppg - 10) + rng.normal(0, 1, n)
    df = pd.DataFrame({"ppg": ppg, "target_ppg_next": target})
    rtm = regression_to_mean(df, n_buckets=5)
    top, bottom = rtm.iloc[-1], rtm.iloc[0]
    # top bucket falls toward the mean; bottom bucket rises toward it
    assert top["mean_ppg_next"] < top["mean_ppg_n"]
    assert bottom["mean_ppg_next"] > bottom["mean_ppg_n"]
    # shrink ~ 0.5 by construction
    assert 0.35 < top["shrink"] < 0.65
