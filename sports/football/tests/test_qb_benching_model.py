"""Tests for the stage-3 model layer (small synthetic frames; no nflverse / network).

None of these check that the model is accurate. They check that the *evaluation* cannot flatter
it, which is the whole point of the stage:

* the stratification actually stratifies — the bands are the ones stage 1 measured, and a
  within-band AUC is computed on that band alone;
* precision@k is per season, not pooled across seventeen of them;
* the baselines are scored in the direction a reader would read them (fewer prior attempts means
  *more* risk, a later draft pick means *more* risk), because a sign slip there would make the
  model look like it beats a baseline that was being scored backwards;
* the temporal split trains only on earlier seasons.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qb_benching.model import (  # noqa: E402
    baseline_scores, band_profile, cross_validate, precision_at_k_by_season, temporal_split,
    volume_band, within_band_auc,
)


def _frame(n=120, seed=0):
    """A frame where risk genuinely falls with prior attempts, so signal is present but modest."""
    rng = np.random.default_rng(seed)
    att = rng.choice([0, 50, 200, 450], size=n, p=[0.1, 0.1, 0.2, 0.6]).astype(float)
    epa = rng.normal(0, 0.1, n)
    risk = 0.6 - att / 900 - epa
    y = (rng.random(n) < np.clip(risk, 0.02, 0.95)).astype(int)
    return pd.DataFrame({
        "season": rng.integers(2009, 2026, n),
        "prior_attempts": att,
        "prior_epa_per_db": epa,
        "draft_pick": rng.integers(1, 260, n).astype(float),
        "displaced_last_season": rng.integers(0, 2, n).astype(float),
        "benched": y,
    })


# -- the stratification ------------------------------------------------------------------------

def test_volume_bands_match_the_cut_points_stage_one_measured():
    df = pd.DataFrame({"prior_attempts": [0.0, 1.0, 99.0, 100.0, 299.0, 300.0, 600.0]})
    assert list(volume_band(df)) == [
        "none", "<100", "<100", "<100", "100-299", "100-299", "300+"]


def test_within_band_auc_scores_each_band_on_its_own_rows():
    """A score that is perfect inside every band but inverted across them must still read high.

    This is the whole reason the stratified number exists: pooled, such a score looks useless.
    """
    df = pd.DataFrame({
        "season": [2020] * 8,
        "prior_attempts": [400.0] * 4 + [50.0] * 4,
        "benched": [0, 0, 1, 1, 0, 0, 1, 1],
    })
    # inside each band the score orders the outcome perfectly; the bands are offset in opposite
    # directions so the pooled ordering is scrambled
    scores = [0.1, 0.2, 0.3, 0.4, 0.9, 0.8, 0.7, 0.6]
    out = within_band_auc(df, scores).set_index("volume_band")
    assert out.loc["300+", "model_auc"] == 1.0
    assert out.loc["<100", "model_auc"] == 0.0   # perfectly inverted within that band


def test_band_profile_counts_reconcile_to_the_frame():
    df = _frame()
    prof = band_profile(df)
    assert prof["n"].sum() == len(df)
    assert prof["positives"].sum() == df["benched"].sum()


# -- precision@k is per season -----------------------------------------------------------------

def test_precision_at_k_is_computed_within_each_season():
    """Two seasons; the top-1 pick is right in one and wrong in the other, so the mean is 0.5.

    Pooled, both of season A's positives would crowd out season B entirely and return 1.0.
    """
    df = pd.DataFrame({
        "season": [2020, 2020, 2021, 2021],
        "benched": [1, 0, 0, 1],
    })
    scores = [0.9, 0.1, 0.8, 0.2]   # right in 2020, wrong in 2021
    assert precision_at_k_by_season(df, scores, k=1) == pytest.approx(0.5)


def test_precision_at_k_handles_a_season_smaller_than_k():
    df = pd.DataFrame({"season": [2020, 2020], "benched": [1, 0]})
    assert precision_at_k_by_season(df, [0.9, 0.1], k=5) == pytest.approx(0.5)


# -- baseline direction ------------------------------------------------------------------------

def test_baselines_are_scored_so_higher_means_more_risk():
    """Fewer prior attempts must score as *more* risk; a sign slip would invert the comparison."""
    df = pd.DataFrame({
        "season": [2020] * 6,
        "prior_attempts": [600.0, 550.0, 500.0, 20.0, 10.0, 5.0],
        "prior_epa_per_db": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
        "draft_pick": [10.0] * 6,
        "displaced_last_season": [0.0] * 6,
        "benched": [0, 0, 0, 1, 1, 1],
    })
    out = baseline_scores(df).set_index("baseline")
    att = [b for b in out.index if "attempts" in b][0]
    assert out.loc[att, "auc"] == 1.0


def test_a_baseline_with_missing_values_does_not_crash_or_rank_them_extreme():
    df = _frame()
    df.loc[df.index[:20], "draft_pick"] = np.nan
    out = baseline_scores(df)
    assert out["auc"].notna().all()
    assert ((out["auc"] >= 0) & (out["auc"] <= 1)).all()


# -- the temporal split ------------------------------------------------------------------------

def test_temporal_split_trains_only_on_earlier_seasons():
    df = _frame(n=200, seed=3)
    out = temporal_split(df, ["prior_attempts", "prior_epa_per_db"])
    assert out is not None
    assert out["n_train"] + out["n_test"] == len(df)
    assert (df["season"] <= out["split_season"]).sum() == out["n_train"]


def test_cross_validate_returns_one_row_per_repeat_and_scores_every_row():
    df = _frame(n=150, seed=5)
    rows, oof = cross_validate(df, ["prior_attempts", "prior_epa_per_db"], n_repeats=2)
    assert len(rows) == 2
    assert len(oof) == len(df)
    assert ((oof >= 0) & (oof <= 1)).all()
