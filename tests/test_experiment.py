"""Tests for Stage 8 experiment harness helpers (pure; synthetic frames)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from position_predictor.eval.experiment import (  # noqa: E402
    TARGET,
    _aggregate_ranking,
    _all_feature_columns,
    _dedupe,
    _score_over_folds,
    _test_label_seasons,
)


def test_dedupe_and_all_feature_columns():
    assert _dedupe(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]
    blocks = {"b1": ["x", "y"], "b2": ["y", "z", "missing"]}
    assert _all_feature_columns(blocks, available={"x", "y", "z"}) == ["x", "y", "z"]


def test_test_label_seasons():
    df = pd.DataFrame({"season": [2015, 2016, 2017, 2018, 2019],
                       TARGET: [1.0, 1.0, 1.0, np.nan, 1.0]})
    # label season = feature season + 1; 2018 has no target so its label (2019) is excluded
    assert _test_label_seasons(df, 3) == [2017, 2018, 2020]


def test_score_over_folds_eligibility_and_keys():
    # two feature seasons (-> labels 2021, 2022); cutoff filters by eligible flag
    rows = []
    for fs, label in [(2020, 2021), (2021, 2022)]:
        for i in range(10):
            rows.append(dict(season=fs, target=float(i),
                             eligible_next__g4=int(i >= 3), eligible_next__g8=int(i >= 6)))
    df = pd.DataFrame(rows)

    def predict_fn(test_rows):  # perfect predictor
        return test_rows[TARGET].to_numpy()

    out = _score_over_folds(predict_fn, df, [2021, 2022], horizon=1,
                            cutoff_grid=[4, 8], k_tiers=(2,),
                            key={"model": "m", "window_years": 10})
    res = pd.DataFrame(out)
    # 2 folds x 2 cutoffs = 4 rows; keys carried through
    assert len(res) == 4
    assert set(res["cutoff_games"]) == {4, 8}
    assert set(res["test_season"]) == {2021, 2022}
    assert (res["spearman"] > 0.99).all()             # perfect predictor
    # g4 keeps i>=3 (7 rows), g8 keeps i>=6 (4 rows)
    assert set(res[res.cutoff_games == 4]["n"]) == {7}
    assert set(res[res.cutoff_games == 8]["n"]) == {4}


def test_aggregate_ranking_mean_sd():
    base = dict(sport="football", position="RB", model_type="era_ensemble", model="m",
                window_years=10, combine="mean", cutoff_games=4)
    df = pd.DataFrame([
        {**base, "test_season": 2021, "mae": 2.0, "rmse": 2.0, "r2": 0.5, "spearman": 0.6,
         "kendall": 0.4, "ndcg_at_12": 0.8, "precision_at_12": 0.5},
        {**base, "test_season": 2022, "mae": 4.0, "rmse": 4.0, "r2": 0.7, "spearman": 0.8,
         "kendall": 0.6, "ndcg_at_12": 0.9, "precision_at_12": 0.7},
    ])
    agg = _aggregate_ranking(df)
    assert len(agg) == 1
    row = agg.iloc[0]
    assert abs(row["spearman_mean"] - 0.7) < 1e-9
    assert abs(row["mae_mean"] - 3.0) < 1e-9
    assert row["n_folds"] == 2
