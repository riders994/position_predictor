"""Expectation models: what a club's injury outcomes *should* have looked like.

Every grade in this project is observed minus expected, so this module builds the expected side.
Three choices carry it, and each is a deliberate rejection of the more obvious option.

**A penalised logistic hazard, not a gradient-boosted tree.** The estimand is a *residual*, and a
flexible learner eats residual variance: a boosted model adjusting on roster-composition proxies
will absorb precisely the between-club variation this project is trying to measure. Penalised
logistic regression with splines on the continuous covariates is the more complex thing that is
*less* dangerous here. A shallow GBM runs as a robustness check only, to show the residuals do
not move.

**Leave-one-team-out, not random k-fold.** Club T is scored by a model fit on the other 31. A
random fold containing some of T's rows lets the expectation absorb T's own effect and
understates it — the same concern that made the sibling project's draft expectation out-of-fold.
:func:`fit_predict_loto` is tested by injecting a synthetic club effect and asserting LOTO
recovers it while random-fold shrinks it.

**Team identity is never a feature.** Not as a column, not as a proxy. If it were, a club could
explain away its own residual. There is a test asserting it is absent from the design matrix.

Discrete-time hazard handles the pervasive censoring (season ends, trades, releases) exactly by
construction, with no extra dependency — for weekly data it is equivalent to Cox.
"""

from __future__ import annotations

import numpy as np

RANDOM_STATE = 17

# Splines on the covariates whose risk is plainly non-linear. Age in particular is U-shaped
# against injury once workload is held fixed, and a linear term would launder that into the
# residual the grades are read from.
SPLINE_COLUMNS = ("age", "week")
N_SPLINE_KNOTS = 5


def build_hazard_pipeline(numeric, categorical, *, C: float = 1.0, spline=None):
    """Impute -> spline the non-linear numerics -> scale -> one-hot -> penalised logistic."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

    spline = [c for c in (spline if spline is not None else SPLINE_COLUMNS) if c in numeric]
    plain = [c for c in numeric if c not in spline]

    blocks = []
    if plain:
        blocks.append(("num", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]), plain))
    if spline:
        blocks.append(("spl", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("spline", SplineTransformer(n_knots=N_SPLINE_KNOTS, degree=3,
                                         include_bias=False)),
        ]), spline))
    if categorical:
        blocks.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=25,
                                     sparse_output=False)),
        ]), categorical))

    return Pipeline([
        ("features", ColumnTransformer(blocks, remainder="drop")),
        ("model", LogisticRegression(C=C, max_iter=2000, solver="lbfgs",
                                     random_state=RANDOM_STATE)),
    ])


def feature_columns(frame, numeric, categorical, *, team_col: str = "team"):
    """The design-matrix columns, with a hard guarantee that club identity is not among them."""
    cols = [c for c in (*numeric, *categorical) if c in frame.columns]
    leaked = [c for c in cols if c == team_col or c.startswith(f"{team_col}_")]
    if leaked:
        raise ValueError(f"club identity leaked into the design matrix: {leaked}")
    return cols


def fit_predict_loto(frame, numeric, categorical, outcome: str, *, team_col: str = "team",
                     C: float = 1.0):
    """Out-of-fold expected probabilities, holding out one club at a time.

    Returns an array aligned to ``frame``'s row order. A club with no training data outside
    itself (never happens with 32 clubs, but guarded) falls back to the pooled base rate.
    """
    numeric = [c for c in numeric if c in frame.columns]
    categorical = [c for c in categorical if c in frame.columns]
    feature_columns(frame, numeric, categorical, team_col=team_col)

    X = frame[[*numeric, *categorical]].to_pandas() if hasattr(frame, "to_pandas") else \
        frame[[*numeric, *categorical]]
    y = np.asarray(frame[outcome]).astype(int)
    teams = np.asarray(frame[team_col])

    out = np.full(len(y), np.nan)
    for club in np.unique(teams):
        test = teams == club
        train = ~test
        if train.sum() == 0 or len(np.unique(y[train])) < 2:
            out[test] = y[train].mean() if train.sum() else y.mean()
            continue
        pipe = build_hazard_pipeline(numeric, categorical, C=C)
        pipe.fit(X[train], y[train])
        out[test] = pipe.predict_proba(X[test])[:, 1]
    return out


def fit_predict_kfold(frame, numeric, categorical, outcome: str, *, n_splits: int = 5,
                      C: float = 1.0):
    """Random k-fold out-of-fold predictions — the *wrong* scheme, kept for the contrast test."""
    from sklearn.model_selection import StratifiedKFold

    numeric = [c for c in numeric if c in frame.columns]
    categorical = [c for c in categorical if c in frame.columns]
    X = frame[[*numeric, *categorical]].to_pandas() if hasattr(frame, "to_pandas") else \
        frame[[*numeric, *categorical]]
    y = np.asarray(frame[outcome]).astype(int)

    out = np.full(len(y), np.nan)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    for train, test in splitter.split(X, y):
        pipe = build_hazard_pipeline(numeric, categorical, C=C)
        pipe.fit(X.iloc[train], y[train])
        out[test] = pipe.predict_proba(X.iloc[test])[:, 1]
    return out


def team_observed_expected(frame, expected, *, team_col: str = "team", outcome: str,
                           extra_group=None):
    """Observed minus expected counts per club (optionally within another grouping)."""
    import pandas as pd

    df = frame.to_pandas() if hasattr(frame, "to_pandas") else frame.copy()
    df = df.assign(_expected=expected, _observed=np.asarray(df[outcome]).astype(int))
    keys = [team_col] + list(extra_group or [])
    agg = df.groupby(keys, dropna=False).agg(
        n=("_observed", "size"),
        observed=("_observed", "sum"),
        expected=("_expected", "sum"),
        # Poisson-binomial variance: independent Bernoulli trials with differing p.
        var_indep=("_expected", lambda s: float(np.sum(s * (1 - s)))),
    ).reset_index()
    agg["diff"] = agg["observed"] - agg["expected"]
    agg["rate_obs"] = agg["observed"] / agg["n"]
    agg["rate_exp"] = agg["expected"] / agg["n"]
    agg["z_indep"] = agg["diff"] / np.sqrt(agg["var_indep"].clip(lower=1e-9))
    return pd.DataFrame(agg)


def poisson_binomial_null(p, *, n_sim: int = 20000, seed: int = RANDOM_STATE):
    """Simulated null total for one club: independent Bernoulli draws at its own fitted p."""
    rng = np.random.default_rng(seed)
    p = np.asarray(p, dtype=float)
    return rng.binomial(1, p[None, :], size=(n_sim, len(p))).sum(axis=1)


def player_block_null(p, blocks, *, n_sim: int = 5000, seed: int = RANDOM_STATE):
    """Resample whole players, each carrying his own vector of fitted probabilities.

    Independent Bernoulli understates variance because a fragile player is fragile all season and
    his weeks are correlated. Resampling blocks keeps that correlation, so the interval is wider
    and the resulting p-values more conservative. The project uses whichever of the two nulls is
    more conservative.
    """
    rng = np.random.default_rng(seed)
    p = np.asarray(p, dtype=float)
    # Blocks arrive as player ids and may carry nulls, which makes a mixed float/str object
    # array that will not sort. Cast to string so every row lands in some block — a null id
    # pooled into one block is a wider (more conservative) null, never a narrower one.
    blocks = np.asarray(blocks, dtype=object)
    blocks = np.array(["__null__" if b is None or b != b else str(b) for b in blocks])
    order = np.argsort(blocks, kind="stable")
    _, starts = np.unique(blocks[order], return_index=True)
    groups = np.split(order, starts[1:])
    if not groups:
        return np.zeros(n_sim)

    totals = np.empty(n_sim)
    n_groups = len(groups)
    for i in range(n_sim):
        picked = rng.integers(0, n_groups, size=n_groups)
        probs = np.concatenate([p[groups[j]] for j in picked])
        totals[i] = rng.binomial(1, probs).sum()
    return totals


def two_sided_p(observed: float, null_draws) -> float:
    """Two-sided p on |observed - mean(null)|, with the +1 correction."""
    null_draws = np.asarray(null_draws, dtype=float)
    centre = null_draws.mean()
    extreme = np.abs(null_draws - centre) >= abs(observed - centre)
    return float((extreme.sum() + 1) / (len(null_draws) + 1))


def whole_factor_permutation(frame, expected, *, team_col: str = "team",
                             unit_col: str = "gsis_id", outcome: str,
                             n_sim: int = 2000, seed: int = RANDOM_STATE):
    """Does club identity explain **any** variance in the residual? The primary inference.

    One test rather than 32, so there is no multiplicity problem. The null reassigns whole
    *players* to clubs — preserving each player's own correlated run of weeks — and asks how often
    the between-club variance of observed-minus-expected is as large as the real one.
    """
    import pandas as pd

    df = frame.to_pandas() if hasattr(frame, "to_pandas") else frame.copy()
    df = df.assign(_e=expected, _o=np.asarray(df[outcome]).astype(int))

    per_unit = df.groupby(unit_col, dropna=False).agg(
        _o=("_o", "sum"), _e=("_e", "sum")).reset_index()
    unit_team = df.groupby(unit_col, dropna=False)[team_col].first().reset_index()
    merged = per_unit.merge(unit_team, on=unit_col)

    def between_var(team_labels):
        tmp = pd.DataFrame({"t": team_labels, "d": merged["_o"] - merged["_e"]})
        return float(tmp.groupby("t")["d"].sum().var(ddof=1))

    actual = between_var(merged[team_col].to_numpy())
    rng = np.random.default_rng(seed)
    labels = merged[team_col].to_numpy()
    draws = np.empty(n_sim)
    for i in range(n_sim):
        draws[i] = between_var(rng.permutation(labels))
    return {"statistic": actual, "null_mean": float(draws.mean()),
            "p_value": float(((draws >= actual).sum() + 1) / (n_sim + 1)),
            "null": draws}


def overdispersion(diff, var_indep):
    """Method-of-moments variance decomposition of the club residuals.

    ``tau2`` is the between-club variance that survives after subtracting the sampling variance
    the model already explains; ``intraclass`` is its share of the total. This is the number that
    reframes a raw spread — it says how much of the observed range is candidate signal rather
    than the luck of who got hurt.
    """
    diff = np.asarray(diff, dtype=float)
    var_indep = np.asarray(var_indep, dtype=float)
    observed_var = float(np.var(diff, ddof=1)) if len(diff) > 1 else 0.0
    mean_sampling = float(np.mean(var_indep))
    tau2 = max(0.0, observed_var - mean_sampling)
    total = tau2 + mean_sampling
    return {"observed_var": observed_var, "mean_sampling_var": mean_sampling, "tau2": tau2,
            "intraclass": (tau2 / total) if total > 0 else 0.0,
            "sd_signal": float(np.sqrt(tau2))}


def detectable_effect(var_indep, *, alpha: float = 0.05):
    """Smallest observed-minus-expected a typical club could have shown and been detected.

    Reported in the outcome's own units (episodes, returns, recurrences). When the grades turn
    out not to be distinguishable, this is the headline — "nothing this size or smaller was
    findable" is a result; "no effect" is not.
    """
    from scipy import stats

    var_indep = np.asarray(var_indep, dtype=float)
    z = stats.norm.ppf(1 - alpha / 2)
    typical_sd = float(np.sqrt(np.median(var_indep)))
    return {"z": float(z), "typical_sd": typical_sd, "min_detectable": float(z * typical_sd)}


__all__ = [
    "N_SPLINE_KNOTS", "RANDOM_STATE", "SPLINE_COLUMNS",
    "build_hazard_pipeline", "detectable_effect", "feature_columns", "fit_predict_kfold",
    "fit_predict_loto", "overdispersion", "player_block_null", "poisson_binomial_null",
    "team_observed_expected", "two_sided_p", "whole_factor_permutation",
]
