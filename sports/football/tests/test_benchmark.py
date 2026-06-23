"""Tests for Stage 9 market benchmark + report helpers (pure; synthetic frames)."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from position_predictor.data.benchmark import (  # noqa: E402
    latest_preseason_by_season,
)
from position_predictor.eval.experiment import _attach_market, _score_benchmark  # noqa: E402
from position_predictor.eval.report import _fmt  # noqa: E402


def _ecr():
    # two scrape dates in 2021 (June = off-window, Sept = preseason) + an off-window 2022
    rows = []
    for i in range(40):
        rows.append(dict(player=f"P{i}", id=str(i), pos="RB", ecr=float(i + 1),
                         ecr_type="ro", scrape_date="2021-06-01"))      # too early
        rows.append(dict(player=f"P{i}", id=str(i), pos="RB", ecr=float(i + 2),
                         ecr_type="ro", scrape_date="2021-09-08"))      # preseason -> picked
    return pd.DataFrame(rows)


def test_latest_preseason_picks_the_in_window_scrape():
    snap = latest_preseason_by_season(_ecr(), min_players=30)
    assert set(snap["season"]) == {2021}
    # the kept scrape is the Sept one (ecr = i+2), not the June one
    assert snap["scrape_date"].dt.month.unique().tolist() == [9]
    assert len(snap) == 40


def test_latest_preseason_skips_thin_scrapes():
    thin = pd.DataFrame([dict(player="A", id="1", pos="RB", ecr=1.0, ecr_type="ro",
                              scrape_date="2021-09-08")])
    assert latest_preseason_by_season(thin, min_players=30).empty


def test_attach_market_and_score_benchmark():
    # feature seasons 2020 (-> label 2021); market keyed by label season
    df = pd.DataFrame([
        dict(player_id=f"P{i}", season=2020, target=float(20 - i),
             eligible_next__g4=1) for i in range(10)])
    # write a market frame to a temp external dir via monkey-free path: call _score directly
    df = df.assign(market_ecr=[float(i + 1) for i in range(10)])  # rank i+1 ~ inverse of target
    rows = _score_benchmark(df, [2021], horizon=1, cutoff_grid=[4], k_tiers=(3,),
                            base_key={"sport": "football", "position": "RB"})
    assert len(rows) == 1
    r = rows[0]
    assert r["model"] == "market_ecr"
    assert r["n_market_ranked"] == 10 and r["coverage"] == 1.0
    # ECR ranks players in the exact reverse of target value -> Spearman ~ +1
    # (lower ecr = better = higher target), so −ecr correlates positively with target
    assert r["spearman"] > 0.99


def test_attach_market_missing_file_is_safe(tmp_path):
    df = pd.DataFrame([dict(player_id="A", season=2020, target=1.0)])
    out = _attach_market(df, stem="does_not_exist_xyz", horizon=1)
    assert "market_ecr" in out.columns and out["market_ecr"].isna().all()


def test_report_fmt():
    assert _fmt(0.12345) == "0.123"
    assert _fmt(None) == "—"
    assert _fmt(1.5, 1) == "1.5"
