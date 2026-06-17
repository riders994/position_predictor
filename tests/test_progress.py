"""Tests for the cross-version progress report helpers (pure; synthetic snapshots)."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from position_predictor.eval.progress import _best_market_row, _version_key  # noqa: E402


def test_version_key_orders_numerically():
    versions = ["v10", "v2", "v1", "v3"]
    assert sorted(versions, key=_version_key) == ["v1", "v2", "v3", "v10"]


def test_best_market_row_separates_market_and_aggregates_folds(tmp_path):
    # two folds per model; means should be taken across folds, market split out.
    pd.DataFrame({
        "model": ["xgboost", "xgboost", "ridge", "ridge", "market_ecr", "market_ecr"],
        "spearman": [0.70, 0.74, 0.60, 0.62, 0.73, 0.75],
        "weighted_tau": [0.65, 0.67, 0.55, 0.57, 0.68, 0.70],
        "precision_at_12": [0.5, 0.6, 0.4, 0.5, 0.55, 0.65],
        "precision_at_24": [0.7, 0.7, 0.6, 0.6, 0.75, 0.75],
    }).to_csv(tmp_path / "benchmark_comparison.csv", index=False)

    best, market = _best_market_row(tmp_path)
    assert best["model"] == "xgboost"          # higher mean spearman than ridge
    assert best["spearman"] == 0.72             # mean of 0.70, 0.74
    assert market["spearman"] == 0.74           # mean of 0.73, 0.75


def test_best_market_row_missing_file_is_safe(tmp_path):
    assert _best_market_row(tmp_path) == (None, None)
