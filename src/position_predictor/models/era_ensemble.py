"""Stage 8 — the era ensemble (PROJECT_PLAN §6.3, §7.3).

One estimator is fit **per era** on that era's nested feature schema, and the per-era predictions
are combined into a single window prediction. Because schemas are nested, every era model can
score a modern (full-feature) test row using its own column subset — so the combiner is a direct
test of whether older-era relationships still predict (the recency-bias study, §6.2).

Combiners (config ``era_modeling.combine``):
- ``val_weighted`` (default) — weight each era model by how well it ranks a held-out **validation**
  tail of the training window (the data decides how much older eras help).
- ``mean`` — equal weight across the era models in the window.
- ``recency_weighted`` — fixed schedule favouring newer eras.

The ensemble is trained on whatever rows the caller passes (the window's training rows); it
partitions them by era internally via ``assign_era`` (route_by feature season N, §6.3).
"""

from __future__ import annotations

from ..eras import assign_era, feature_columns_for_era
from .zoo import make_estimator


def _era_columns(era, block_columns, available):
    # Blocks overlap (e.g. ``receptions`` is in both production and volume), so dedupe while
    # preserving first-seen order — a model matrix must not repeat a feature column.
    cols, seen = [], set()
    for c in feature_columns_for_era(era, block_columns):
        if c in available and c not in seen:
            cols.append(c)
            seen.add(c)
    return cols


def _fit_one(estimator_name, rows, cols, target_col, seed):
    est = make_estimator(estimator_name, seed=seed)
    est.fit(rows[cols], rows[target_col])
    return est


class EraEnsemble:
    """Fit-per-era estimator with a configurable window combiner (§7.3)."""

    def __init__(self, estimator_name, eras, block_columns, *, combine="val_weighted",
                 val_seasons=3, target_col="target", season_col="season", seed=1729,
                 weight_metric="spearman"):
        self.estimator_name = estimator_name
        self.eras = eras
        self.block_columns = block_columns
        self.combine = combine
        self.val_seasons = val_seasons
        self.target_col = target_col
        self.season_col = season_col
        self.seed = seed
        self.weight_metric = weight_metric
        # populated by fit()
        self.models_ = {}      # era name -> fitted estimator
        self.columns_ = {}     # era name -> feature columns used
        self.weights_ = {}     # era name -> combine weight (sum to 1)
        self.eras_present_ = []

    # ----------------------------------------------------------------- fit helpers
    def _present_eras(self, df):
        names = {assign_era(int(s), self.eras) for s in df[self.season_col].unique()}
        return [e for e in self.eras if e.name in names]

    def _val_weights(self, df, present):
        """Weight ∝ each era model's mean within-season Spearman on a held-out val tail."""
        import numpy as np

        from ..eval.metrics import ranking_metrics

        seasons = sorted(df[self.season_col].unique())
        if len(seasons) <= self.val_seasons:
            return None  # too few seasons to carve a validation tail -> caller falls back
        val_seasons = set(seasons[-self.val_seasons:])
        fit_df = df[~df[self.season_col].isin(val_seasons)]
        val_df = df[df[self.season_col].isin(val_seasons)]
        scores = {}
        for era in present:
            cols = _era_columns(era, self.block_columns, df.columns)
            era_fit = fit_df[fit_df[self.season_col].map(
                lambda s: assign_era(int(s), self.eras)) == era.name]
            if not cols or len(era_fit) < 10:
                continue
            est = _fit_one(self.estimator_name, era_fit, cols, self.target_col, self.seed)
            per_season = []
            for _, g in val_df.groupby(self.season_col):
                pred = est.predict(g[cols])
                per_season.append(ranking_metrics(
                    g[self.target_col], pred)[self.weight_metric])
            s = float(np.nanmean(per_season)) if per_season else np.nan
            scores[era.name] = s
        if not scores or all(np.isnan(v) for v in scores.values()):
            return None
        # eras with a usable score get max(score, 0); eras without get the mean of the rest
        good = {k: max(v, 0.0) for k, v in scores.items() if v == v}
        mean_good = (sum(good.values()) / len(good)) if good else 0.0
        raw = {e.name: good.get(e.name, mean_good) for e in present}
        total = sum(raw.values())
        if total <= 0:
            return None
        return {k: v / total for k, v in raw.items()}

    def _recency_weights(self, present):
        # newer eras (later start_season) weigh more, linearly by rank.
        order = sorted(present, key=lambda e: e.start_season)
        ranks = {e.name: i + 1 for i, e in enumerate(order)}
        total = sum(ranks.values())
        return {k: v / total for k, v in ranks.items()}

    # ------------------------------------------------------------------------- fit
    def fit(self, train_df):
        df = train_df.dropna(subset=[self.target_col])
        present = self._present_eras(df)
        self.eras_present_ = [e.name for e in present]

        weights = None
        if self.combine == "val_weighted":
            weights = self._val_weights(df, present)
        elif self.combine == "recency_weighted":
            weights = self._recency_weights(present)
        if weights is None:  # 'mean', or val_weighted with too little data -> equal weights
            weights = {e.name: 1.0 / len(present) for e in present}

        # Final models: refit each era model on ALL its rows in the window.
        for era in present:
            cols = _era_columns(era, self.block_columns, df.columns)
            era_rows = df[df[self.season_col].map(
                lambda s: assign_era(int(s), self.eras)) == era.name]
            if not cols or era_rows.empty:
                continue
            self.models_[era.name] = _fit_one(
                self.estimator_name, era_rows, cols, self.target_col, self.seed)
            self.columns_[era.name] = cols

        # Restrict weights to eras that actually produced a model and renormalise.
        w = {k: weights.get(k, 0.0) for k in self.models_}
        total = sum(w.values())
        self.weights_ = {k: (v / total if total > 0 else 1.0 / len(w)) for k, v in w.items()}
        return self

    # --------------------------------------------------------------------- predict
    def predict(self, test_df):
        import numpy as np
        if not self.models_:
            raise RuntimeError("EraEnsemble is not fitted")
        out = np.zeros(len(test_df), dtype=float)
        for name, est in self.models_.items():
            cols = self.columns_[name]
            out += self.weights_[name] * np.asarray(est.predict(test_df[cols]), dtype=float)
        return out
