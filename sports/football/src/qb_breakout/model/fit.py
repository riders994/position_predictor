"""Fitting and — mostly — refusing to over-claim from 35 positives.

At this sample size the modelling choices matter less than the evaluation ones. A flexible learner
will reach a cross-validated AUC of 0.8 on 178 rows and mean nothing by it, so this module is
built around three guards:

**A regularised linear model, not a forest.** L2 logistic regression with a small feature set is
the most complex thing 35 positives support. Gradient boosting is fitted too, purely so the report
can show it does not help.

**A label-permutation null.** Every headline number is accompanied by the distribution of that
same number with the outcome shuffled. This is the only honest reference point when a single fold
holds seven positives: an AUC of 0.65 means something quite different if shuffled labels reach it
a fifth of the time.

**A temporal split as well as a random one.** Random cross-validation lets the model learn from
2019 to predict 2009, which is not how it would ever be used. The temporal split trains on early
entrants and tests on later ones — the deployment simulation, and the number to believe when the
two disagree.
"""

from __future__ import annotations

RANDOM_STATE = 17
N_SPLITS = 5
N_REPEATS = 20
N_PERMUTATIONS = 500

# The practical use of this model is a shortlist, not a probability. Precision at k answers the
# question a reader actually has: of the k quarterbacks it likes most, how many broke out?
TOP_K = 15


def build_pipeline(numeric, categorical, *, kind="logistic", random_state=RANDOM_STATE):
    """Impute, standardise, one-hot, then fit. Imputation is median and inside the fold."""
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    num = Pipeline([("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"))])
    pre = ColumnTransformer([("num", num, list(numeric)), ("cat", cat, list(categorical))])

    if kind == "logistic":
        # C=0.1 is a deliberate, strong shrink: at 35 positives an unregularised fit produces
        # coefficients that flip sign between folds.
        model = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced",
                                   random_state=random_state)
    elif kind == "boosting":
        model = HistGradientBoostingClassifier(max_depth=2, max_iter=120, learning_rate=0.05,
                                               random_state=random_state)
    else:
        raise ValueError(f"unknown model {kind!r}")
    return Pipeline([("pre", pre), ("model", model)])


def precision_at_k(y_true, scores, k=TOP_K):
    import numpy as np

    order = np.argsort(-np.asarray(scores))[:k]
    return float(np.asarray(y_true)[order].mean())


def cross_validate(frame, numeric, categorical, outcome="ever_sustained", *, kind="logistic",
                   n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE):
    """Repeated stratified CV, returning out-of-fold scores per repeat.

    Out-of-fold predictions are kept per repeat rather than pooled across repeats, because pooling
    averages away exactly the fold-to-fold instability that is the headline fact at this N.
    """
    import numpy as np
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    X = frame[list(numeric) + list(categorical)]
    y = frame[outcome].to_numpy()

    rows, oof_last = [], None
    for repeat in range(n_repeats):
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state + repeat)
        oof = np.zeros(len(y), dtype=float)
        for train_idx, test_idx in cv.split(X, y):
            pipe = build_pipeline(numeric, categorical, kind=kind, random_state=random_state)
            pipe.fit(X.iloc[train_idx], y[train_idx])
            oof[test_idx] = pipe.predict_proba(X.iloc[test_idx])[:, 1]
        rows.append({
            "repeat": repeat,
            "auc": roc_auc_score(y, oof),
            "brier": brier_score_loss(y, oof),
            "precision_at_k": precision_at_k(y, oof),
        })
        oof_last = oof
    return rows, oof_last


def permutation_null(frame, numeric, categorical, outcome="ever_sustained", *, kind="logistic",
                     n_permutations=N_PERMUTATIONS, n_splits=N_SPLITS, random_state=RANDOM_STATE):
    """Cross-validated AUC with the labels shuffled — what "no signal" actually looks like here.

    Returns the array of shuffled AUCs; the caller compares its observed value against it.
    Shuffling happens *outside* the CV loop, so the whole pipeline including imputation is re-run
    against noise exactly as it is against the real labels.
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    X = frame[list(numeric) + list(categorical)]
    y = frame[outcome].to_numpy()
    rng = np.random.default_rng(random_state)

    aucs = []
    for i in range(n_permutations):
        y_perm = rng.permutation(y)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state + i)
        oof = np.zeros(len(y), dtype=float)
        for train_idx, test_idx in cv.split(X, y_perm):
            pipe = build_pipeline(numeric, categorical, kind=kind, random_state=random_state)
            pipe.fit(X.iloc[train_idx], y_perm[train_idx])
            oof[test_idx] = pipe.predict_proba(X.iloc[test_idx])[:, 1]
        aucs.append(roc_auc_score(y_perm, oof))
    return np.asarray(aucs)


def temporal_split(frame, numeric, categorical, outcome="ever_sustained", *, kind="logistic",
                   split_season=None, random_state=RANDOM_STATE):
    """Train on earlier NFL entrants, test on later ones — the deployment simulation.

    Random folds let the model learn from 2019 to predict 2009. This is the only evaluation whose
    information flow matches how the model would actually be used, so where it disagrees with the
    random-fold number, this one is the one to believe.
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score

    seasons = frame["entry_season"]
    if split_season is None:
        split_season = int(np.median(seasons))

    train = frame[seasons <= split_season]
    test = frame[seasons > split_season]
    cols = list(numeric) + list(categorical)
    if test[outcome].nunique() < 2 or train[outcome].nunique() < 2:
        return None

    pipe = build_pipeline(numeric, categorical, kind=kind, random_state=random_state)
    pipe.fit(train[cols], train[outcome])
    scores = pipe.predict_proba(test[cols])[:, 1]
    return {
        "split_season": int(split_season),
        "n_train": int(len(train)), "train_positives": int(train[outcome].sum()),
        "n_test": int(len(test)), "test_positives": int(test[outcome].sum()),
        "auc": float(roc_auc_score(test[outcome], scores)),
        "precision_at_k": precision_at_k(test[outcome], scores, k=min(TOP_K, len(test))),
    }


def benchmark_scores(frame, outcome="ever_sustained"):
    """The draft, as an independent opinion — never blended into the model.

    Scored so that a higher number means a better prospect, i.e. the negated overall pick, with
    undrafted players placed behind everyone drafted.
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score

    pick = frame["draft_pick"].to_numpy(dtype=float)
    worst = np.nanmax(pick) + 1 if np.isfinite(pick).any() else 1.0
    scores = -np.where(np.isnan(pick), worst, pick)
    y = frame[outcome].to_numpy()
    return {
        "auc": float(roc_auc_score(y, scores)),
        "precision_at_k": precision_at_k(y, scores),
        "n": int(len(y)), "positives": int(y.sum()),
    }


# Round-ish bands by overall pick. Coarse on purpose: finer bands would put single-digit positive
# counts in every cell.
DRAFT_BANDS = ((0, 32, "R1"), (32, 105, "R2-3"), (105, 10_000, "R4+/UDFA"))


def draft_band(frame):
    """Label each quarterback by draft band, with undrafted players in the last one."""
    import numpy as np

    pick = frame["draft_pick"].to_numpy(dtype=float)
    pick = np.where(np.isnan(pick), 10_000 - 1, pick)
    labels = np.empty(len(pick), dtype=object)
    for lo, hi, name in DRAFT_BANDS:
        labels[(pick > lo) & (pick <= hi)] = name
    return labels


def opportunity_profile(frame, outcome="ever_sustained"):
    """Breakout rate and NFL playing time by draft band.

    The draft is a strong predictor of this outcome, and this is the table that shows why to be
    careful about that: a quarterback drafted in the fourth round barely plays, and a fantasy
    breakout is impossible without snaps. The draft partly *causes* the outcome it appears to
    predict, so beating it is not the standard a college-only model should be held to.
    """

    df = frame.copy()
    df["draft_band"] = draft_band(df)
    cols = {"n": (outcome, "size"), "breakouts": (outcome, "sum"), "rate": (outcome, "mean")}
    if "career_games" in df.columns:
        cols["mean_nfl_games"] = ("career_games", "mean")
    if "career_starts" in df.columns:
        cols["mean_nfl_starts"] = ("career_starts", "mean")
    out = df.groupby("draft_band", observed=True).agg(**cols).reset_index()
    order = [name for _, _, name in DRAFT_BANDS]
    out["_o"] = out["draft_band"].map({n: i for i, n in enumerate(order)})
    return out.sort_values("_o").drop(columns="_o").round(3)


def within_band_auc(frame, scores, outcome="ever_sustained"):
    """Does college production separate breakouts *among similarly-drafted* quarterbacks?

    This is the comparison that matters. Globally a college model and the draft agree because both
    reward the same production, so a global AUC mostly re-reports the draft. Holding the draft band
    fixed asks whether the college evidence knows anything the league did not — which is precisely
    the late-breakout question, since late breakouts are quarterbacks the league undervalued.

    The draft's own within-band ordering is reported alongside, as an independent opinion.
    """
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    df = frame.copy()
    df["draft_band"] = draft_band(df)
    df["_score"] = np.asarray(scores)
    pick = df["draft_pick"].to_numpy(dtype=float)
    df["_draft_score"] = -np.where(np.isnan(pick), np.nanmax(pick) + 1, pick)

    rows = []
    for band in [name for _, _, name in DRAFT_BANDS]:
        grp = df[df["draft_band"] == band]
        if len(grp) == 0 or grp[outcome].nunique() < 2:
            rows.append({"draft_band": band, "n": len(grp),
                         "breakouts": int(grp[outcome].sum()) if len(grp) else 0,
                         "model_auc": None, "draft_auc": None})
            continue
        rows.append({
            "draft_band": band, "n": len(grp), "breakouts": int(grp[outcome].sum()),
            "model_auc": round(float(roc_auc_score(grp[outcome], grp["_score"])), 3),
            "draft_auc": round(float(roc_auc_score(grp[outcome], grp["_draft_score"])), 3),
        })
    return pd.DataFrame(rows)


def coefficients(frame, numeric, categorical, outcome="ever_sustained", *,
                 random_state=RANDOM_STATE):
    """Coefficients of a fit on everything — for reading direction, not for inference.

    A coefficient from a single fit on 178 rows is a description of this sample. The report pairs
    each one with how often its sign survives resampling, which is the part worth reading.
    """
    import numpy as np
    import pandas as pd

    cols = list(numeric) + list(categorical)
    pipe = build_pipeline(numeric, categorical, random_state=random_state)
    pipe.fit(frame[cols], frame[outcome])
    names = pipe.named_steps["pre"].get_feature_names_out()
    coefs = pipe.named_steps["model"].coef_[0]

    rng = np.random.default_rng(random_state)
    signs = np.zeros((40, len(coefs)))
    for i in range(40):
        idx = rng.choice(len(frame), len(frame), replace=True)
        boot = frame.iloc[idx]
        if boot[outcome].nunique() < 2:
            signs[i] = np.nan
            continue
        p = build_pipeline(numeric, categorical, random_state=random_state)
        p.fit(boot[cols], boot[outcome])
        signs[i] = np.sign(p.named_steps["model"].coef_[0])

    agreement = np.nanmean(signs == np.sign(coefs), axis=0)
    return pd.DataFrame({
        "feature": [n.split("__", 1)[-1] for n in names],
        "coef": np.round(coefs, 3),
        "sign_stability": np.round(agreement, 2),
    }).sort_values("coef", key=abs, ascending=False).reset_index(drop=True)
