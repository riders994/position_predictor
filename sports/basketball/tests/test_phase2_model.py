"""Tests for the Phase-2 composition->success model (pure helpers + a planted-signal sanity check)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nba_archetypes.eval.phase2_model import (  # noqa: E402
    compare_models, feature_cols, fit_coefficients, quartile_contrast)


def _synthetic(beta=0.6, noise=0.01, seed=0):
    """8 leagues x 10 teams; cat_win_rate driven (with strength ``beta``) by comp_A's share."""
    rng = np.random.default_rng(seed)
    rows = []
    for lg in range(8):
        shares = rng.dirichlet([2, 2, 2, 2], size=10)  # 4 compositional archetypes summing to 1
        for t in range(10):
            a, b, c, d = shares[t]
            y = 0.5 + beta * (a - 0.25) + rng.normal(0, noise)
            rows.append({"league_key": f"L{lg}", "team_key": f"L{lg}.t{t}",
                         "comp_A": a, "comp_B": b, "comp_C": c, "comp_D": d, "cat_win_rate": y})
    return pd.DataFrame(rows)


def test_feature_cols_picks_comp_columns():
    df = _synthetic()
    assert feature_cols(df) == ["comp_A", "comp_B", "comp_C", "comp_D"]


def test_compare_models_recovers_planted_signal_out_of_fold():
    # With a real linear signal, Ridge must beat the mean baseline out-of-fold (leave-one-league-out).
    df = _synthetic(beta=0.6, noise=0.01)
    res = compare_models(df)
    assert res["baseline (mean)"]["oof_r2"] <= 0.01          # mean baseline ~0 by construction
    assert res["Ridge"]["oof_r2"] > 0.5                      # strong planted signal is recovered
    assert res["Ridge"]["oof_mae"] < res["baseline (mean)"]["oof_mae"]


def test_compare_models_no_signal_is_near_zero():
    # Pure noise target -> no model should show positive out-of-fold R² (guards against leakage).
    df = _synthetic(beta=0.0, noise=0.05)
    res = compare_models(df)
    assert res["Ridge"]["oof_r2"] < 0.1


def test_fit_coefficients_sign_and_stability():
    df = _synthetic(beta=0.6, noise=0.01)
    coefs = fit_coefficients(df, n_boot=200)
    top = coefs[0]                                            # largest positive coefficient
    assert top["archetype"] == "A"
    assert top["coef_std"] > 0 and top["sign_stability"] >= 0.95
    assert {r["archetype"] for r in coefs} == {"A", "B", "C", "D"}


def test_quartile_contrast_orders_by_diff():
    df = _synthetic(beta=0.6, noise=0.01)
    contrast = quartile_contrast(df)
    assert contrast[0]["archetype"] == "A"                   # winners hold more comp_A
    assert contrast[0]["diff"] > 0
    diffs = [r["diff"] for r in contrast]
    assert diffs == sorted(diffs, reverse=True)
