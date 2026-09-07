"""Fitting, and being careful about which number is the result.

90 positives in 544 rows is a workable sample — far more than ``qb_breakout`` had — but the
evaluation choices still matter more than the modelling ones, and one of them decides whether
this project has a finding at all.

**The stratified AUC is the headline, not the pooled one.** Stage 1's descriptive pass showed
that benching rate runs 47.4% for an opener with under 100 prior attempts against 11.6% for one
over 300. A model that learns only "this man was never really the starter" will post a strong
pooled AUC and tell a reader nothing they did not already know from the depth chart. This is the
same trap ``qb_breakout`` documented, where draft position scored 0.890 pooled and 0.496 within a
band, because capital *allocates* the opportunity rather than predicting the outcome. So every
headline is reported **within prior-volume strata**, and the within-stratum number is the one to
believe.

**A label-permutation null.** Every AUC is quoted against the distribution of the same statistic
with the outcome shuffled, so "0.65" can be read against what noise reaches.

**A temporal split as well as a random one.** Random folds let 2024 inform a prediction about
2009. The temporal split trains on early seasons and tests on late ones — the deployment
simulation, and the number to believe where the two disagree.

**Baselines that must be beaten.** Each is a single column a reader already has. A model that
cannot beat prior attempts is not adding anything.
"""

from __future__ import annotations

RANDOM_STATE = 17
N_SPLITS = 5
N_REPEATS = 20
N_PERMUTATIONS = 500

#: Openers flagged per season. Roughly the number actually benched in a year (90 over 17 seasons
#: is 5.3), so it is the shortlist a reader would really draw.
TOP_K_PER_SEASON = 5

#: Prior-season attempt bands — the "was he ever really the starter" axis, and the stratification
#: the headline is computed within. Cut points follow stage 1's descriptive table.
VOLUME_BANDS = ((-1, 0, "none"), (0, 100, "<100"), (100, 300, "100-299"), (300, 10_000, "300+"))


def build_pipeline(numeric, categorical=(), *, kind="logistic", random_state=RANDOM_STATE):
    """Impute, standardise, one-hot, then fit. Imputation is median and inside the fold."""
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    num = Pipeline([("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler())])
    steps = [("num", num, list(numeric))]
    if len(categorical):
        cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"))])
        steps.append(("cat", cat, list(categorical)))
    pre = ColumnTransformer(steps)

    if kind == "logistic":
        # A deliberate shrink. Several features here are near-collinear by construction (attempts,
        # dropbacks and starts all measure the same thing), and an unregularised fit splits the
        # weight between them differently in every fold.
        model = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced",
                                   random_state=random_state)
    elif kind == "boosting":
        model = HistGradientBoostingClassifier(max_depth=2, max_iter=150, learning_rate=0.05,
                                               random_state=random_state)
    else:
        raise ValueError(f"unknown model {kind!r}")
    return Pipeline([("pre", pre), ("model", model)])


def volume_band(frame):
    """Label each opener by prior-season attempts."""
    import numpy as np

    att = frame["prior_attempts"].to_numpy(dtype=float)
    labels = np.empty(len(att), dtype=object)
    for lo, hi, name in VOLUME_BANDS:
        labels[(att > lo) & (att <= hi)] = name
    return labels


def precision_at_k_by_season(frame, scores, outcome="benched", k=TOP_K_PER_SEASON):
    """Mean precision of the ``k`` riskiest openers in each season.

    Pooled precision@k is misleading here because the pool spans seventeen drafts; a reader picks
    from one season's thirty-two starters. This is that question.
    """
    import numpy as np
    import pandas as pd

    df = pd.DataFrame({"season": frame["season"].to_numpy(),
                       "y": frame[outcome].to_numpy().astype(float),
                       "s": np.asarray(scores)})
    hits = []
    for _, grp in df.groupby("season"):
        top = grp.nlargest(min(k, len(grp)), "s")
        hits.append(top["y"].mean())
    return float(np.mean(hits))


def cross_validate(frame, numeric, categorical=(), outcome="benched", *, kind="logistic",
                   n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE):
    """Repeated stratified CV, returning per-repeat metrics and the last out-of-fold scores.

    Out-of-fold predictions are kept per repeat rather than pooled, because pooling averages away
    the fold-to-fold instability that is worth reporting.
    """
    import numpy as np
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    cols = list(numeric) + list(categorical)
    X = frame[cols]
    y = frame[outcome].to_numpy().astype(int)

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
            "precision_at_k": precision_at_k_by_season(frame, oof, outcome),
        })
        oof_last = oof
    return rows, oof_last


def permutation_null(frame, numeric, categorical=(), outcome="benched", *, kind="logistic",
                     n_permutations=N_PERMUTATIONS, n_splits=N_SPLITS,
                     random_state=RANDOM_STATE):
    """Cross-validated AUC with the labels shuffled — what "no signal" looks like at this N."""
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    cols = list(numeric) + list(categorical)
    X = frame[cols]
    y = frame[outcome].to_numpy().astype(int)
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


def temporal_split(frame, numeric, categorical=(), outcome="benched", *, kind="logistic",
                   split_season=None, random_state=RANDOM_STATE):
    """Train on earlier seasons, test on later ones — the deployment simulation."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    seasons = frame["season"]
    if split_season is None:
        split_season = int(np.median(seasons))
    train, test = frame[seasons <= split_season], frame[seasons > split_season]
    if train[outcome].nunique() < 2 or test[outcome].nunique() < 2:
        return None

    cols = list(numeric) + list(categorical)
    pipe = build_pipeline(numeric, categorical, kind=kind, random_state=random_state)
    pipe.fit(train[cols], train[outcome].astype(int))
    scores = pipe.predict_proba(test[cols])[:, 1]
    return {
        "split_season": int(split_season),
        "n_train": int(len(train)), "train_positives": int(train[outcome].sum()),
        "n_test": int(len(test)), "test_positives": int(test[outcome].sum()),
        "auc": float(roc_auc_score(test[outcome].astype(int), scores)),
        "precision_at_k": precision_at_k_by_season(test, scores, outcome),
    }


#: Single columns a reader already has. ``sign`` is +1 when a larger value means more risk.
BASELINES = (
    ("prior_attempts", -1, "prior-season attempts (the 'was he really the starter' axis)"),
    ("prior_epa_per_db", -1, "prior-season EPA per dropback"),
    ("draft_pick", +1, "draft capital (later pick = more risk)"),
    ("displaced_last_season", +1, "lost the job last season"),
)


def baseline_scores(frame, outcome="benched"):
    """Each single-column baseline, scored the way a reader would read it."""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    y = frame[outcome].to_numpy().astype(int)
    rows = []
    for col, sign, label in BASELINES:
        if col not in frame.columns:
            continue
        raw = frame[col].to_numpy(dtype=float)
        # a missing value is the least informative position, not the most extreme one
        filled = np.where(np.isnan(raw), np.nanmedian(raw), raw)
        scores = sign * filled
        rows.append({
            "baseline": label,
            "auc": round(float(roc_auc_score(y, scores)), 3),
            "precision_at_k": round(precision_at_k_by_season(frame, scores, outcome), 3),
        })
    return pd.DataFrame(rows)


def band_profile(frame, outcome="benched"):
    """Benching rate and sample size per prior-volume band — why stratification is needed."""
    df = frame.copy()
    df["volume_band"] = volume_band(df)
    out = df.groupby("volume_band", observed=True).agg(
        n=(outcome, "size"), positives=(outcome, "sum"), rate=(outcome, "mean")).reset_index()
    order = {name: i for i, (_, _, name) in enumerate(VOLUME_BANDS)}
    return out.assign(_o=out["volume_band"].map(order)).sort_values("_o").drop(
        columns="_o").round(3).reset_index(drop=True)


def within_band_auc(frame, scores, outcome="benched"):
    """Does the model separate benchings *among similarly-established* starters?

    The comparison that decides whether this project has a finding. Pooled, the model and the
    depth chart agree because both know a stopgap when they see one. Holding prior volume fixed
    asks whether the preseason evidence knows anything beyond that.
    """
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    df = frame.copy()
    df["volume_band"] = volume_band(df)
    df["_score"] = np.asarray(scores)

    rows = []
    for _, _, band in VOLUME_BANDS:
        grp = df[df["volume_band"] == band]
        if len(grp) == 0 or grp[outcome].nunique() < 2:
            rows.append({"volume_band": band, "n": len(grp),
                         "positives": int(grp[outcome].sum()) if len(grp) else 0,
                         "model_auc": None})
            continue
        rows.append({
            "volume_band": band, "n": len(grp), "positives": int(grp[outcome].sum()),
            "model_auc": round(float(roc_auc_score(grp[outcome].astype(int), grp["_score"])), 3),
        })
    return pd.DataFrame(rows)


def coefficients(frame, numeric, categorical=(), outcome="benched", *,
                 random_state=RANDOM_STATE, n_boot=60):
    """Coefficients from a fit on everything, paired with how often each sign survives a bootstrap.

    A single fit's coefficient describes this sample. The sign-stability column is the part worth
    reading.
    """
    import numpy as np
    import pandas as pd

    cols = list(numeric) + list(categorical)
    pipe = build_pipeline(numeric, categorical, random_state=random_state)
    y = frame[outcome].astype(int)
    pipe.fit(frame[cols], y)
    names = pipe.named_steps["pre"].get_feature_names_out()
    coefs = pipe.named_steps["model"].coef_[0]

    rng = np.random.default_rng(random_state)
    signs = np.zeros((n_boot, len(coefs)))
    for i in range(n_boot):
        idx = rng.choice(len(frame), len(frame), replace=True)
        boot = frame.iloc[idx]
        if boot[outcome].nunique() < 2:
            signs[i] = np.nan
            continue
        p = build_pipeline(numeric, categorical, random_state=random_state)
        p.fit(boot[cols], boot[outcome].astype(int))
        signs[i] = np.sign(p.named_steps["model"].coef_[0])

    agreement = np.nanmean(signs == np.sign(coefs), axis=0)
    return pd.DataFrame({
        "feature": [n.split("__", 1)[-1] for n in names],
        "coef": np.round(coefs, 3),
        "sign_stability": np.round(agreement, 2),
    }).sort_values("coef", key=abs, ascending=False).reset_index(drop=True)
