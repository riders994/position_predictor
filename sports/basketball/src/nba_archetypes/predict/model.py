"""Phase 3 — baselines + **walk-forward** evaluation of the next-season archetype predictor.

The must-beat baseline is **persistence** ("same archetype as last season") — archetypes are sticky
(YoY ~0.63), so the model earns its keep only by beating that both on hard top-1 accuracy *and* on the
soft membership distribution (log-loss/Brier). Evaluation is **walk-forward**: to score transitions into
season ``s`` we train only on transitions into seasons ``< s`` (never peeking at the future), then pool
the out-of-fold predictions. All baselines are scored on exactly the pooled evaluated rows so the
comparison is apples-to-apples.
"""
from __future__ import annotations

import numpy as np

from .build import feature_columns, target_prob_columns

N_ARCH = 12
LABELS = list(range(N_ARCH))


def _aligned_proba(est, X):
    """predict_proba re-indexed to the full 0..11 label space (0 for classes absent from training)."""
    proba = est.predict_proba(X)
    full = np.zeros((len(X), N_ARCH))
    full[:, est.classes_] = proba
    return full


def _metrics(y_true, proba, soft_target=None):
    """Top-1 accuracy, macro-F1, log-loss, and (soft) multiclass Brier vs the N+1 membership vector."""
    from sklearn.metrics import accuracy_score, f1_score, log_loss

    pred = proba.argmax(axis=1)
    eps = 1e-12
    p = np.clip(proba, eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    out = {"accuracy": round(float(accuracy_score(y_true, pred)), 4),
           "macro_f1": round(float(f1_score(y_true, pred, average="macro", labels=LABELS,
                                             zero_division=0)), 4),
           "log_loss": round(float(log_loss(y_true, p, labels=LABELS)), 4)}
    if soft_target is not None:
        out["brier"] = round(float(((p - soft_target) ** 2).sum(axis=1).mean()), 4)
    return out


def _make_estimator(seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0,
        early_stopping=False, random_state=seed)


def walk_forward(table, *, kind="yoe", min_train_seasons=2, seed=1729):
    """Walk-forward evaluation: predict transitions into each season from only-earlier transitions.

    ``kind`` selects the feature set (``'yoe'`` = Model A, ``'age'`` = Model B). Returns a dict with the
    pooled model metrics, the persistence and marginal baselines on the same rows, the model's top-1 lift
    over persistence, and the count of evaluated pairs.
    """
    cols = feature_columns(table, kind=kind)
    tp = target_prob_columns(table)
    df = table.copy()
    df["target_season"] = df["season"] + 1
    target_seasons = sorted(df["target_season"].unique())
    eval_from = target_seasons[min_train_seasons:]        # need >=min_train_seasons earlier target years

    model_proba, pers_proba, y_all, soft_all, marg_pred = [], [], [], [], []
    for s in eval_from:
        tr = df[df["target_season"] < s]
        te = df[df["target_season"] == s]
        if te.empty or tr["target_arch"].nunique() < 2:
            continue
        est = _make_estimator(seed)
        est.fit(tr[cols].to_numpy(dtype=float), tr["target_arch"].to_numpy())
        model_proba.append(_aligned_proba(est, te[cols].to_numpy(dtype=float)))
        # persistence as a SOFT predictor = the player's current membership vector (p0..p11)
        pcols = table.attrs.get("pcols") or [f"p{j}" for j in range(N_ARCH)]
        pers_proba.append(te[pcols].to_numpy(dtype=float))
        marg_pred.append(np.full(len(te), tr["target_arch"].mode().iloc[0]))
        y_all.append(te["target_arch"].to_numpy())
        soft_all.append(te[tp].to_numpy(dtype=float))

    y = np.concatenate(y_all)
    soft = np.concatenate(soft_all)
    model_proba = np.concatenate(model_proba)
    pers_proba = np.concatenate(pers_proba)
    marg = np.concatenate(marg_pred)

    model_m = _metrics(y, model_proba, soft)
    pers_m = _metrics(y, pers_proba, soft)
    marg_proba = np.zeros((len(y), N_ARCH))
    marg_proba[np.arange(len(y)), marg] = 1.0
    marg_m = _metrics(y, marg_proba, soft)
    return {"model": model_m, "persistence": pers_m, "marginal": marg_m,
            "acc_lift_vs_persistence": round(model_m["accuracy"] - pers_m["accuracy"], 4),
            "logloss_gain_vs_persistence": round(pers_m["log_loss"] - model_m["log_loss"], 4),
            "n_eval": int(len(y)), "eval_seasons": [int(s) for s in eval_from], "kind": kind}


def feature_importance(table, *, kind="yoe", seed=1729, top=15):
    """Permutation importance of the fitted model (whole-table fit) — what drives the prediction."""
    from sklearn.inspection import permutation_importance

    cols = feature_columns(table, kind=kind)
    X = table[cols].to_numpy(dtype=float)
    y = table["target_arch"].to_numpy()
    est = _make_estimator(seed).fit(X, y)
    imp = permutation_importance(est, X, y, n_repeats=5, random_state=seed, scoring="accuracy")
    rows = [{"feature": c, "importance": round(float(imp.importances_mean[i]), 4)}
            for i, c in enumerate(cols)]
    rows.sort(key=lambda r: r["importance"], reverse=True)
    return rows[:top]
