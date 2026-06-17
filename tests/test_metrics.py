"""Tests for Stage 8 scoring metrics (pure; known closed-form values)."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from position_predictor.eval.metrics import (  # noqa: E402
    availability_metrics,
    ndcg_at_k,
    precision_at_k,
    ranking_metrics,
    regression_metrics,
)


def test_regression_metrics_known():
    m = regression_metrics([1, 2, 3], [1, 2, 3])
    assert m["mae"] == 0 and m["rmse"] == 0 and m["r2"] == 1.0
    m2 = regression_metrics([0, 0, 0, 0], [1, -1, 1, -1])
    assert abs(m2["mae"] - 1.0) < 1e-9 and abs(m2["rmse"] - 1.0) < 1e-9


def test_regression_ignores_nan_pairs():
    m = regression_metrics([1, 2, np.nan, 4], [1, 2, 3, np.nan])
    assert m["n"] == 2  # only the two fully-finite pairs count


def test_precision_at_k():
    # perfect agreement on the top-2 -> 1.0
    assert precision_at_k([10, 9, 1, 2], [5, 4, 0, 1], k=2) == 1.0
    # top-2 predicted = {idx0, idx3}; top-2 true = {idx0, idx1} -> overlap 1 -> 0.5
    assert precision_at_k([10, 9, 1, 8], [10, 1, 2, 9], k=2) == 0.5
    # k larger than n clamps to n
    assert precision_at_k([3, 1], [1, 3], k=5) == 1.0


def test_ndcg_at_k_perfect_and_reversed():
    yt = [3, 2, 1, 0]
    assert abs(ndcg_at_k(yt, [3, 2, 1, 0], k=4) - 1.0) < 1e-9   # perfect order
    # reversed order is worse than perfect
    assert ndcg_at_k(yt, [0, 1, 2, 3], k=4) < 1.0


def test_ranking_metrics_monotonic():
    r = ranking_metrics([1, 2, 3, 4, 5], [2, 4, 6, 8, 10], k_tiers=(2,))
    assert abs(r["spearman"] - 1.0) < 1e-9
    assert abs(r["kendall"] - 1.0) < 1e-9
    assert abs(r["weighted_tau"] - 1.0) < 1e-9
    assert r["precision_at_2"] == 1.0


def test_weighted_tau_penalizes_top_more_than_bottom():
    true = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    swap_top = [9, 10, 8, 7, 6, 5, 4, 3, 2, 1]      # swap the best two
    swap_bottom = [10, 9, 8, 7, 6, 5, 4, 3, 1, 2]   # swap the worst two
    wt_top = ranking_metrics(true, swap_top, k_tiers=(2,))["weighted_tau"]
    wt_bot = ranking_metrics(true, swap_bottom, k_tiers=(2,))["weighted_tau"]
    # a top swap hurts the top-weighted score much more than a bottom swap
    assert wt_top < wt_bot
    # plain Spearman is indifferent to where the swap happens
    sp_top = ranking_metrics(true, swap_top, k_tiers=(2,))["spearman"]
    sp_bot = ranking_metrics(true, swap_bottom, k_tiers=(2,))["spearman"]
    assert abs(sp_top - sp_bot) < 1e-9


def test_availability_metrics_auc_and_single_class():
    # games predictor perfectly ranks clears-cutoff (>=4) -> AUC 1.0
    yt = [0, 2, 5, 10]
    yp = [0.1, 1.0, 6.0, 9.0]
    m = availability_metrics(yt, yp, cutoff=4)
    assert m["clears_auc"] == 1.0
    assert m["games_mae"] >= 0
    # single-class target -> AUC NaN, no crash
    m2 = availability_metrics([5, 6, 7], [5, 6, 7], cutoff=4)
    assert np.isnan(m2["clears_auc"])
