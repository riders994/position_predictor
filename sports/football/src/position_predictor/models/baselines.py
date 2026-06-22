"""Stage 8 — must-beat baselines (PROJECT_PLAN §7.1).

Each baseline is a function ``(train_df, test_df) -> np.ndarray`` predicting next-season PPG for
the ``test_df`` rows, fit only on ``train_df`` (no leakage). They are era-agnostic — they need
only the player's season-*N* PPG (and, for ``linear``, a small core feature set) — so they form
the floor the era ensemble (§7.3) must clear at every window.
"""

from __future__ import annotations

# Small, always-available core for the linear baseline (subset present across all eras).
LINEAR_CORE = ["ppg", "touches_pg", "age", "target_share", "carries_pg", "finish_ppg_rank"]


def persistence(train_df, test_df, *, target_col="target", seed=0):
    """Next-season PPG = this-season PPG (the naive carry-forward)."""
    import numpy as np
    return np.asarray(test_df["ppg"], dtype=float)


def smoothed_history(train_df, test_df, *, target_col="target", seed=0, w_recent=0.6):
    """Weighted blend of this season and the reconstructed prior season's PPG.

    Prior PPG is recovered from ``ppg - ppg_delta1`` (the YoY delta the feature stage already
    computes from *prior* seasons); rows without a prior fall back to this season's PPG.
    """
    import numpy as np
    ppg = np.asarray(test_df["ppg"], dtype=float)
    if "ppg_delta1" in test_df.columns:
        prior = ppg - np.asarray(test_df["ppg_delta1"], dtype=float)
        prior = np.where(np.isfinite(prior), prior, ppg)
        return w_recent * ppg + (1 - w_recent) * prior
    return ppg


def mean_reversion(train_df, test_df, *, target_col="target", seed=0):
    """Regress this-season PPG toward the training population's next-season mean.

    Estimates the shrink slope ``b = cov(ppg, target)/var(ppg)`` on the training rows; the
    prediction ``mean_target + b·(ppg − mean_ppg_train)`` pulls extreme seasons toward the mean
    (``b < 1`` ⇒ reversion), exactly the EDA's regression-to-mean read (Stage 6).
    """
    import numpy as np
    tr = train_df[["ppg", target_col]].dropna()
    ppg_test = np.asarray(test_df["ppg"], dtype=float)
    if len(tr) < 2:
        return ppg_test
    x = tr["ppg"].to_numpy(dtype=float)
    y = tr[target_col].to_numpy(dtype=float)
    var = x.var()
    b = float(np.cov(x, y)[0, 1] / var) if var > 0 else 0.0
    return y.mean() + b * (ppg_test - x.mean())


def linear(train_df, test_df, *, target_col="target", seed=0, cols=None):
    """Ordinary least squares on a small core feature set (median-imputed, train-fit)."""
    import numpy as np
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline

    cols = [c for c in (cols or LINEAR_CORE) if c in train_df.columns]
    tr = train_df.dropna(subset=[target_col])
    model = make_pipeline(SimpleImputer(strategy="median"), LinearRegression())
    model.fit(tr[cols], tr[target_col])
    return np.asarray(model.predict(test_df[cols]), dtype=float)


BASELINES = {
    "persistence": persistence,
    "smoothed_history": smoothed_history,
    "mean_reversion": mean_reversion,
    "linear": linear,
}
