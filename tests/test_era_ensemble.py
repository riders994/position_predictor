"""Tests for Stage 8 era ensemble + combiners (synthetic data with a known signal)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from position_predictor.eras import Era  # noqa: E402
from position_predictor.models.era_ensemble import EraEnsemble, _era_columns  # noqa: E402

ERAS = [
    Era("boxscore", 2000, 2007, ("base",)),
    Era("snaps", 2008, 2013, ("base", "snap")),
    Era("ngs", 2014, None, ("base", "snap", "ngs")),
]
BLOCKS = {"base": ["x", "x"], "snap": ["s"], "ngs": ["g"]}  # note duplicate x across calls


def _synthetic(seed=0):
    """target = 2*x + noise; one row-block per season across all three eras."""
    rng = np.random.default_rng(seed)
    rows = []
    for season in range(2000, 2018):
        for _ in range(40):
            x = rng.normal(0, 1)
            s = rng.normal(0, 1)
            g = rng.normal(0, 1)
            rows.append(dict(season=season, x=x, s=s, g=g, target=2 * x + rng.normal(0, 0.3)))
    return pd.DataFrame(rows)


def test_era_columns_dedupe_and_filter():
    cols = _era_columns(ERAS[2], {"base": ["x", "x", "y"], "ngs": ["g", "missing"]},
                        available={"x", "y", "g"})
    assert cols == ["x", "y", "g"]  # deduped, missing dropped, order preserved


def test_fit_predict_beats_mean_and_weights_normalised():
    df = _synthetic()
    train = df[df.season <= 2014]
    test = df[df.season >= 2015]
    ens = EraEnsemble("ridge", ERAS, BLOCKS, combine="val_weighted",
                      target_col="target", seed=1).fit(train)
    # weights over the eras present sum to 1
    assert abs(sum(ens.weights_.values()) - 1.0) < 1e-9
    assert set(ens.weights_) <= {"boxscore", "snaps", "ngs"}
    pred = ens.predict(test)
    assert len(pred) == len(test)
    # learned signal -> predictions correlate strongly with the held-out target
    assert np.corrcoef(pred, test["target"])[0, 1] > 0.9


def test_combiners_all_run_and_normalise():
    df = _synthetic(1)
    train = df[df.season <= 2014]
    for combine in ("val_weighted", "mean", "recency_weighted"):
        ens = EraEnsemble("ridge", ERAS, BLOCKS, combine=combine,
                          target_col="target", seed=1).fit(train)
        assert abs(sum(ens.weights_.values()) - 1.0) < 1e-9
    # mean -> equal weights across the present eras
    ens_mean = EraEnsemble("ridge", ERAS, BLOCKS, combine="mean",
                           target_col="target", seed=1).fit(train)
    w = list(ens_mean.weights_.values())
    assert max(w) - min(w) < 1e-9
    # recency -> newest era weighted strictly more than the oldest
    ens_rec = EraEnsemble("ridge", ERAS, BLOCKS, combine="recency_weighted",
                          target_col="target", seed=1).fit(train)
    assert ens_rec.weights_["ngs"] > ens_rec.weights_["boxscore"]
