"""Stage 8 — candidate estimator factory (PROJECT_PLAN §7.2, §7.4).

Returns unfitted scikit-learn-compatible regressors by name. Two NaN-handling regimes match the
plan (§7.3): **linear** families and the sklearn **random forest** get a median-impute (+ scale
for linear) pipeline, while **LightGBM / XGBoost** consume raw NaN via their native split
handling. Imputation is fit inside the pipeline (train-only) so there is no leakage.

Also exposes :func:`make_availability_estimator` — the §7.4 count model for N+1 games played.
``lightgbm``/``xgboost`` are imported lazily so the package imports without them.
"""

from __future__ import annotations

LINEAR_FAMILY = {"ridge", "lasso", "elasticnet"}
TREE_NATIVE_NAN = {"lightgbm", "xgboost"}
CANDIDATES = ["ridge", "lasso", "elasticnet", "random_forest", "lightgbm", "xgboost"]


def make_estimator(name: str, *, seed: int = 1729):
    """Build an unfitted regressor for ``name`` (see :data:`CANDIDATES`)."""
    name = name.lower()
    if name in LINEAR_FAMILY:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import ElasticNet, Lasso, Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        model = {
            "ridge": Ridge(alpha=1.0, random_state=seed),
            "lasso": Lasso(alpha=0.05, random_state=seed, max_iter=5000),
            "elasticnet": ElasticNet(alpha=0.05, l1_ratio=0.5, random_state=seed,
                                     max_iter=5000),
        }[name]
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), model)

    if name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline

        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                  random_state=seed, n_jobs=-1))

    if name == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=31,
                             min_child_samples=20, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, random_state=seed, n_jobs=-1, verbose=-1)

    if name == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=4,
                            subsample=0.8, colsample_bytree=0.8, random_state=seed,
                            n_jobs=-1, tree_method="hist", verbosity=0)

    raise ValueError(f"unknown estimator '{name}' (known: {CANDIDATES})")


def make_availability_estimator(*, seed: int = 1729):
    """Count model for N+1 games played (§7.4): a Poisson-objective gradient booster.

    LightGBM with the ``poisson`` objective handles the 0–17 count target and raw NaN natively;
    its baseline is the player's prior-season games (handled in the experiment harness).
    """
    from lightgbm import LGBMRegressor
    return LGBMRegressor(objective="poisson", n_estimators=300, learning_rate=0.03,
                         num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1,
                         colsample_bytree=0.8, random_state=seed, n_jobs=-1, verbose=-1)
