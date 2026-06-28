"""Phase 2 — model: **archetype composition -> 9-cat success** (which mixes win).

PROJECT_PLAN §4 deliverable #2. The unit is a fantasy **team-season**; features are the 12 archetype
soft shares (``comp_*``, weeks-weighted, sum≈1); the target is **category-win rate** over the regular
season (the user's chosen success label). The corpus is the ~88 Yahoo redraft team-seasons.

Three things drive the design, all from the data:

1. **Small, collinear, compositional.** n≈88 with 12 shares that sum to 1 (one is redundant), so we
   **regularize** (Ridge) and read coefficients as *relative tilts* away from the average roster mix —
   not independent partial effects. Standardized coefficients give the "shift weight toward archetype
   X" direction; a CLR transform is a noted future refinement (some shares are exact 0, breaking logs).
2. **The label is ~zero-sum within a league** (each league's cat-win rates average 0.5). So honest
   evaluation holds out **whole league-seasons** (``GroupKFold``, leave-one-league-out) — predicting
   one team from its 9 leaguemates would leak. Out-of-fold R²/MAE vs a mean baseline is the real test.
3. **Weak marginal signal is a finding, not a bug.** 9-cat success is a *portfolio* property; with
   n≈88 a near-zero out-of-fold R² is the honest answer, and the value is the *direction* (coefficient
   signs + their bootstrap stability) and the top-vs-bottom composition contrast, not point prediction.
"""
from __future__ import annotations

import numpy as np

from ..utils.io import DATA_PROCESSED, read_parquet

COMP_PREFIX = "comp_"


def load_table(path=None):
    """Load the Phase-2 composition table (Yahoo by default)."""
    return read_parquet(path or DATA_PROCESSED / "phase2_yahoo_composition.parquet")


def feature_cols(df):
    """The archetype soft-share feature columns (``comp_*``)."""
    return [c for c in df.columns if c.startswith(COMP_PREFIX)]


def _groupkfold_oof(estimator, X, y, groups):
    """Leave-one-group-out out-of-fold predictions (one fold per unique group)."""
    from sklearn.base import clone
    from sklearn.model_selection import GroupKFold

    groups = np.asarray(groups)
    n_splits = len(np.unique(groups))
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        est = clone(estimator)
        est.fit(X[tr], y[tr])
        oof[te] = est.predict(X[te])
    return oof


def _r2_mae(y, pred):
    from sklearn.metrics import mean_absolute_error, r2_score
    return float(r2_score(y, pred)), float(mean_absolute_error(y, pred))


def compare_models(df, *, target="cat_win_rate", group="league_key", seed=1729):
    """Leave-one-league-out R²/MAE for a mean baseline vs Ridge/Lasso/GBM (honest generalization)."""
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import LassoCV, RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cols = feature_cols(df)
    X = df[cols].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)
    groups = df[group].to_numpy()
    alphas = np.logspace(-3, 3, 25)

    models = {
        "baseline (mean)": DummyRegressor(strategy="mean"),
        "Ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=alphas)),
        "Lasso": make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=seed, max_iter=20000)),
        "GBM (depth2)": GradientBoostingRegressor(max_depth=2, n_estimators=200,
                                                  learning_rate=0.03, subsample=0.8,
                                                  random_state=seed),
    }
    out = {}
    for name, est in models.items():
        oof = _groupkfold_oof(est, X, y, groups)
        r2, mae = _r2_mae(y, oof)
        out[name] = {"oof_r2": round(r2, 4), "oof_mae": round(mae, 4)}
    return out


def fit_coefficients(df, *, target="cat_win_rate", n_boot=2000, group="league_key", seed=1729):
    """Standardized Ridge coefficients (the archetype→success tilt) with group-bootstrap sign stability.

    Returns a list of dicts (one per archetype) sorted by coefficient, each with the standardized
    coefficient and ``sign_stability`` = the fraction of league-resampled bootstrap fits that keep the
    coefficient's sign. With n≈88 this stability matters more than the point estimate.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cols = feature_cols(df)
    X = df[cols].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)
    groups = df[group].to_numpy()
    alphas = np.logspace(-3, 3, 25)

    def _fit(Xi, yi):
        pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))
        pipe.fit(Xi, yi)
        return pipe.named_steps["ridgecv"].coef_

    coef = _fit(X, y)

    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    signs = np.zeros((n_boot, len(cols)))
    for b in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)      # resample whole leagues
        idx = np.concatenate([np.where(groups == g)[0] for g in pick])
        signs[b] = np.sign(_fit(X[idx], y[idx]))
    stability = (np.sign(coef)[None, :] == signs).mean(axis=0)

    rows = [{"archetype": c[len(COMP_PREFIX):], "coef_std": float(coef[j]),
             "sign_stability": round(float(stability[j]), 3)} for j, c in enumerate(cols)]
    rows.sort(key=lambda r: r["coef_std"], reverse=True)
    return rows


def quartile_contrast(df, *, target="cat_win_rate"):
    """Mean composition of the top vs bottom success quartile, and their difference (descriptive)."""
    cols = feature_cols(df)
    q_hi, q_lo = df[target].quantile(0.75), df[target].quantile(0.25)
    top = df[df[target] >= q_hi][cols].mean()
    bot = df[df[target] <= q_lo][cols].mean()
    rows = [{"archetype": c[len(COMP_PREFIX):], "top_q": round(float(top[c]), 3),
             "bottom_q": round(float(bot[c]), 3), "diff": round(float(top[c] - bot[c]), 3)}
            for c in cols]
    rows.sort(key=lambda r: r["diff"], reverse=True)
    return rows
