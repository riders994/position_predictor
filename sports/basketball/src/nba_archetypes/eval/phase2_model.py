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
    """Load the Phase-2 composition table (Yahoo by default) with derived success targets."""
    return derive_targets(read_parquet(path or DATA_PROCESSED / "phase2_yahoo_composition.parquet"))


def derive_targets(df):
    """Add the alternative success targets that aren't stored directly.

    - ``reg_win_pct`` — regular-season H2H record ``(W + 0.5T) / games`` (the actual standings driver,
      less playoff-luck than final rank).
    - ``rank_score`` — final standings **normalized within league** ``(N - rank)/(N - 1)`` so it's
      comparable across 10- and 12-team leagues (champion 1.0, last 0.0; higher = better, like the
      other targets).
    """
    df = df.copy()
    if {"reg_wins", "reg_losses", "reg_ties"}.issubset(df.columns):
        games = df["reg_wins"] + df["reg_losses"] + df["reg_ties"]
        df["reg_win_pct"] = np.where(games > 0,
                                     (df["reg_wins"] + 0.5 * df["reg_ties"]) / games, np.nan)
    if "final_rank" in df.columns:
        n = df.groupby("league_key")["final_rank"].transform("max")
        df["rank_score"] = np.where(n > 1, (n - df["final_rank"]) / (n - 1), np.nan)
    return df


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


def _within_league_spearman(pred, y, groups):
    """Mean within-league Spearman(predicted, actual) — does the model *order* a league's teams right?

    This is the natural skill metric for a ranking/zero-sum target: it ignores absolute scale and only
    asks whether, inside each held-out league, higher-scored rosters really finished better.
    """
    from scipy.stats import spearmanr
    groups = np.asarray(groups)
    vals = []
    for g in np.unique(groups):
        m = groups == g
        if m.sum() > 2 and np.std(pred[m]) > 1e-12:
            rho = spearmanr(pred[m], y[m]).correlation
            if np.isfinite(rho):
                vals.append(rho)
    return float(np.mean(vals)) if vals else float("nan")


def compare_models(df, *, target="cat_win_rate", group="league_key", seed=1729):
    """Leave-one-league-out R²/MAE for a mean baseline vs Ridge/Lasso/GBM (honest generalization)."""
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import LassoCV, RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    df = df[df[target].notna()]
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
        out[name] = {"oof_r2": round(r2, 4), "oof_mae": round(mae, 4),
                     "oof_spearman": round(_within_league_spearman(oof, y, groups), 4)}
    return out


def representation_shootout(df, *, target="sim_cat_win_rate", group="season", seed=1729):
    """Compare success **representations** under one honest Ridge + leave-one-group-out protocol.

    The Phase-2 question is *which features carry the composition→success signal*. On the simulated
    corpus (group = ``season``) we score three feature sets against the same target:

    - **archetype shares** (``comp_*``) — Phase-1's interpretable mix, but discards category info;
    - **prior coverage** (``cov_pri_*``) — the team's season-N-1 9-cat z-profile, **knowable at draft**;
    - **actual coverage** (``cov_act_*``) — the season-N z-profile, the post-hoc ceiling.

    Returns ``{name: {oof_r2, oof_mae, n_features}}``. The expected story: shares ≈ 0 (wrong
    representation), prior coverage small-positive (draft-time projection is the binding constraint),
    actual coverage strongly positive (category coverage *is* the mechanism).
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    df = df[df[target].notna()]
    y = df[target].to_numpy(dtype=float)
    groups = df[group].to_numpy()
    alphas = np.logspace(-3, 3, 25)
    reps = {
        "archetype shares": [c for c in df.columns if c.startswith(COMP_PREFIX)],
        "prior coverage": [c for c in df.columns if c.startswith("cov_pri_")],
        "actual coverage": [c for c in df.columns if c.startswith("cov_act_")],
    }
    out = {}
    for name, cols in reps.items():
        if not cols:
            continue
        X = df[cols].to_numpy(dtype=float)
        est = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))
        oof = _groupkfold_oof(est, X, y, groups)
        r2, mae = _r2_mae(y, oof)
        out[name] = {"oof_r2": round(r2, 4), "oof_mae": round(mae, 4), "n_features": len(cols)}
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

    df = df[df[target].notna()]
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
