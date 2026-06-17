"""Stage 8 — scoring statistics (PROJECT_PLAN §8).

Three metric families, all pure functions over aligned ``(y_true, y_pred)`` arrays:

- **Regression** on predicted PPG: MAE, RMSE, R².
- **Ranking** (the real objective): Spearman ρ, Kendall τ, NDCG@k, Precision@k for the fantasy
  tiers (top-12/24/36). Ranking is computed *within a test season* — players are ranked against
  their season's peers, never across seasons.
- **Availability** (§7.4): games-played MAE/RMSE plus AUC / PR-AUC for the binary
  "clears the games cutoff" classification.

``numpy``/``scipy``/``sklearn`` are imported lazily so importing the package stays cheap.
"""

from __future__ import annotations


def regression_metrics(y_true, y_pred) -> dict:
    """MAE, RMSE, R² for predicted vs actual PPG."""
    import numpy as np

    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]
    if yt.size == 0:
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan, "n": 0}
    err = yp - yt
    ss_res = float((err ** 2).sum())
    ss_tot = float(((yt - yt.mean()) ** 2).sum())
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "r2": (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "n": int(yt.size),
    }


def precision_at_k(y_true, y_pred, k: int) -> float:
    """Share of the top-*k* *predicted* that are truly in the top-*k* by actual value.

    Ties broken by sort order; if fewer than *k* observations exist the denominator is the
    observation count (so it stays in [0, 1]).
    """
    import numpy as np

    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    n = yt.size
    if n == 0:
        return float("nan")
    kk = min(k, n)
    top_pred = set(np.argsort(-yp)[:kk].tolist())
    top_true = set(np.argsort(-yt)[:kk].tolist())
    return len(top_pred & top_true) / kk


def ndcg_at_k(y_true, y_pred, k: int) -> float:
    """NDCG@k with actual PPG as the gain (top-of-board emphasis).

    Uses non-negative gains (PPG is ≥ 0 in practice; clipped for safety).
    """
    import numpy as np

    yt = np.clip(np.asarray(y_true, dtype=float), 0, None)
    yp = np.asarray(y_pred, dtype=float)
    n = yt.size
    if n == 0 or yt.sum() == 0:
        return float("nan")
    kk = min(k, n)

    def _dcg(order):
        gains = yt[order][:kk]
        discounts = 1.0 / np.log2(np.arange(2, kk + 2))
        return float((gains * discounts).sum())

    dcg = _dcg(np.argsort(-yp))
    idcg = _dcg(np.argsort(-yt))
    return dcg / idcg if idcg > 0 else float("nan")


def ranking_metrics(y_true, y_pred, *, k_tiers=(12, 24, 36)) -> dict:
    """Spearman ρ, Kendall τ, **top-weighted** Kendall τ, and NDCG@k / Precision@k per tier.

    ``weighted_tau`` is a rank correlation in [-1, 1] whose pairs are weighted by rank position
    with a hyperbolic decay (rank 1 weighs most): a misranking near the top of the board costs
    far more than the same swap deep down — the "count it less the farther from #1" objective.
    Plain Spearman/Kendall weight every rank pair equally.
    """
    import numpy as np
    from scipy.stats import kendalltau, spearmanr, weightedtau

    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]
    out = {"n": int(yt.size)}
    if yt.size < 2:
        out.update({"spearman": np.nan, "kendall": np.nan, "weighted_tau": np.nan})
        for k in k_tiers:
            out[f"ndcg_at_{k}"] = np.nan
            out[f"precision_at_{k}"] = np.nan
        return out
    rho = spearmanr(yt, yp).correlation
    tau = kendalltau(yt, yp).correlation
    # hyperbolic rank weighting (scipy default): swaps among top-ranked players dominate.
    wtau = weightedtau(yt, yp).correlation
    out["spearman"] = float(rho) if rho == rho else np.nan
    out["kendall"] = float(tau) if tau == tau else np.nan
    out["weighted_tau"] = float(wtau) if wtau == wtau else np.nan
    for k in k_tiers:
        out[f"ndcg_at_{k}"] = ndcg_at_k(yt, yp, k)
        out[f"precision_at_{k}"] = precision_at_k(yt, yp, k)
    return out


def availability_metrics(y_true_games, y_pred_games, *, cutoff: int) -> dict:
    """Games-played MAE/RMSE + AUC & PR-AUC for the 'clears the games cutoff' label.

    ``cutoff`` is the chosen eligibility games threshold (g*). The classification target is
    ``games >= cutoff``; AUC/PR-AUC are NaN when the held-out set is single-class.
    """
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score

    yt = np.asarray(y_true_games, dtype=float)
    yp = np.asarray(y_pred_games, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]
    reg = regression_metrics(yt, yp)
    out = {"games_mae": reg["mae"], "games_rmse": reg["rmse"], "n": reg["n"]}
    clears = (yt >= cutoff).astype(int)
    if clears.min() != clears.max():  # need both classes for AUC/PR-AUC
        out["clears_auc"] = float(roc_auc_score(clears, yp))
        out["clears_pr_auc"] = float(average_precision_score(clears, yp))
    else:
        out["clears_auc"] = np.nan
        out["clears_pr_auc"] = np.nan
    return out
