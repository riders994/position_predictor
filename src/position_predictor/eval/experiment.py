"""Stage 8 — the walk-forward experiment (PROJECT_PLAN §6–§8).

Ties the pieces together into the headline run:

- **Walk-forward, fixed test block (§6.1–6.2).** The last ``test_block_seasons`` *label* seasons
  are held out; for each history window (10/20/30 yr) the training set spans that many feature
  seasons back from ``min(test) − 1 − horizon`` (``eras.train_window_bounds`` — no leakage). Each
  held-out season is a **fold**; metrics are reported per fold and aggregated (mean ± sd).
- **Era ensemble (§7.3).** Every candidate family is fit per era and composed by each combiner
  (``val_weighted`` default, ``mean``, ``recency_weighted``) — reported side by side so the
  recency-bias study reads both *how much* history helps and *how best* to weight it.
- **Baselines (§7.1).** Era-agnostic must-beat floor, refit on each window's training rows.
- **Eligibility sensitivity (§6.4).** The chosen rule is *applied, not refit*, inside folds;
  ranking metrics are reported across the whole candidate games grid for robustness.
- **NGS-block ablation (§7.3).** The ``ngs`` era model is scored with and without the
  ``ngs_efficiency`` block on the high-coverage tiers (P@12/24/36 + Spearman/NDCG).
- **Availability model (§7.4).** A parallel Poisson count model for N+1 games, vs the
  prior-games baseline, scored with games MAE/RMSE and clears-cutoff AUC/PR-AUC.

Results are tidy tables keyed by (sport, position, model, window, combine, cutoff, fold), written
to ``reports/results/``. Logic is importable/testable; ``run_experiment.py`` does the I/O.
"""

from __future__ import annotations

from ..eras import (assign_era, feature_columns_for_era, load_eras,
                    train_window_bounds)
from ..models.baselines import BASELINES
from ..models.era_ensemble import EraEnsemble
from ..models.zoo import make_availability_estimator
from .metrics import availability_metrics, ranking_metrics, regression_metrics

TARGET = "target"
GAMES_TARGET = "games_next"


def _all_feature_columns(block_columns, available):
    """Deduped union of every block's columns present in the frame (full modern schema)."""
    cols, seen = [], set()
    for blk in block_columns.values():
        for c in blk:
            if c in available and c not in seen:
                cols.append(c)
                seen.add(c)
    return cols


def _eval_row(elig_df, pred, k_tiers):
    m = regression_metrics(elig_df[TARGET], pred)
    r = ranking_metrics(elig_df[TARGET], pred, k_tiers=k_tiers)
    r.pop("n", None)
    return {**m, **r}


def _test_label_seasons(df, k):
    label_seasons = sorted(int(s) + 1 for s in df.loc[df[TARGET].notna(), "season"].unique())
    return label_seasons[-k:]


def run_experiment(config, *, write: bool = True, models=None, windows=None,
                   combiners=None, fast: bool = False):
    """Run the full Stage-8 experiment; return a dict of tidy result DataFrames."""
    import json

    import pandas as pd

    from ..utils.io import DATA_PROCESSED, REPORTS_DIR, ensure_dir, read_parquet

    sport = config.get("experiment.sport", "sport")
    position = config.require("experiment.position")
    seed = int(config.get("reproducibility.random_seed", 1729))
    horizon = int(config.get("target.predict_horizon", 1))
    earliest = int(config.get("data.earliest_season", 1999))
    k_block = int(config.get("validation.test_block_seasons", 5))
    windows = windows or config.get("validation.history_windows_years", [10, 20, 30])
    cutoff_grid = config.get("eligibility.candidate_games_played", [4, 6, 8, 10, 12])
    g_star = int(config.get("eligibility.chosen_games_played", 4))
    k_tiers = tuple(config.get("metrics.precision_at_k_tiers", [12, 24, 36]))
    candidates = models or config.get("models.candidates",
                                      ["ridge", "lasso", "elasticnet", "random_forest",
                                       "lightgbm", "xgboost"])
    combiners = combiners or [config.get("era_modeling.combine", "val_weighted"),
                              *config.get("era_modeling.combine_alternatives", [])]
    if fast:  # quick smoke configuration
        candidates = [c for c in ["ridge", "lightgbm"] if c in candidates] or candidates[:2]
        combiners = combiners[:1]
        windows = windows[:2]

    eras = load_eras(config)
    stem = f"{sport}_{position}".lower()
    df = read_parquet(DATA_PROCESSED / f"{stem}_features.parquet").rename(
        columns={"target_ppg_next": TARGET})
    block_columns = json.load(open(DATA_PROCESSED / f"{stem}_feature_blocks.json"))
    df = _attach_market(df, stem, horizon)  # adds 'market_ecr' (NaN where unranked / missing)

    test_labels = _test_label_seasons(df, k_block)
    test_feat_seasons = [s - horizon for s in test_labels]
    base_key = {"sport": sport, "position": position}

    rank_rows, avail_rows, ablation_rows = [], [], []

    for window in windows:
        n_min, n_max = train_window_bounds(window, test_labels, horizon=horizon,
                                           earliest_season=earliest)
        train = df[(df["season"] >= n_min) & (df["season"] <= n_max) & df[TARGET].notna()]
        if train.empty:
            continue

        # ---- era-ensemble candidates × combiners ----
        for combine in combiners:
            for model in candidates:
                ens = EraEnsemble(model, eras, block_columns, combine=combine,
                                  target_col=TARGET, seed=seed).fit(train)
                rank_rows += _score_over_folds(
                    ens.predict, df, test_labels, horizon, cutoff_grid, k_tiers,
                    {**base_key, "model_type": "era_ensemble", "model": model,
                     "window_years": window, "combine": combine,
                     "era_weights": json.dumps(ens.weights_)})

        # ---- baselines (era-agnostic; refit on this window) ----
        for name, fn in BASELINES.items():
            def _predict(test_rows, _fn=fn, _train=train):
                return _fn(_train, test_rows, target_col=TARGET, seed=seed)
            rank_rows += _score_over_folds(
                _predict, df, test_labels, horizon, cutoff_grid, k_tiers,
                {**base_key, "model_type": "baseline", "model": name,
                 "window_years": window, "combine": "n/a", "era_weights": ""})

        # ---- availability model (§7.4) ----
        avail_rows += _availability_over_folds(
            train, df, eras, block_columns, test_labels, horizon, g_star, seed,
            {**base_key, "window_years": window})

        # ---- NGS-block ablation (§7.3) ----
        ablation_rows += _ngs_ablation(
            train, df, eras, block_columns, test_feat_seasons, cutoff_grid, k_tiers,
            g_star, seed, {**base_key, "window_years": window})

    # ---- market benchmark (§7.4): window-independent, scored once ----
    bench_rows = _score_benchmark(df, test_labels, horizon, cutoff_grid, k_tiers, base_key)
    bench_cmp = _benchmark_comparison(
        df, eras, block_columns, test_labels, horizon, max(windows),
        combiners[0], candidates, g_star, k_tiers, earliest, seed, base_key)

    results = {
        "ranking": pd.DataFrame(rank_rows),
        "availability": pd.DataFrame(avail_rows),
        "ngs_ablation": pd.DataFrame(ablation_rows),
        "benchmark": pd.DataFrame(bench_rows),
        "benchmark_comparison": pd.DataFrame(bench_cmp),
    }
    results["ranking_aggregate"] = _aggregate_ranking(results["ranking"])
    results["recency"] = _recency_table(results["ranking_aggregate"])

    if not write:
        return results

    out = ensure_dir(REPORTS_DIR / "results")
    for name, tbl in results.items():
        tbl.to_csv(out / f"experiment_{stem}_{name}.csv", index=False)
    _write_summary(results, base_key, test_labels, windows, g_star, out, stem)
    return results


def _score_over_folds(predict_fn, df, test_labels, horizon, cutoff_grid, k_tiers, key):
    """Predict each held-out season and score across the eligibility grid → tidy rows."""
    rows = []
    for label in test_labels:
        feat_season = label - horizon
        test_all = df[(df["season"] == feat_season) & df[TARGET].notna()]
        if test_all.empty:
            continue
        pred = predict_fn(test_all)
        test_all = test_all.assign(_pred=pred)
        for cutoff in cutoff_grid:
            flag = f"eligible_next__g{cutoff}"
            elig = test_all[test_all[flag] == 1] if flag in test_all else test_all
            if len(elig) < 2:
                continue
            rows.append({**key, "cutoff_games": int(cutoff), "test_season": int(label),
                         **_eval_row(elig, elig["_pred"].to_numpy(), k_tiers)})
    return rows


def _attach_market(df, stem, horizon):
    """Left-join preseason market ECR onto each feature-season row by its **label** season Y."""
    from ..utils.io import DATA_EXTERNAL, read_parquet

    path = DATA_EXTERNAL / f"market_{stem}.parquet"
    if not path.exists():
        df = df.copy()
        df["market_ecr"] = float("nan")
        return df
    market = read_parquet(path)[["player_id", "season", "market_ecr"]].rename(
        columns={"season": "_label"})
    out = df.copy()
    out["_label"] = out["season"] + horizon
    out = out.merge(market, on=["player_id", "_label"], how="left")
    return out.drop(columns="_label")


def _score_benchmark(df, test_labels, horizon, cutoff_grid, k_tiers, base_key):
    """Score the preseason market (−ECR) per fold on the eligible∩ranked universe (+ coverage)."""
    rows = []
    if "market_ecr" not in df or df["market_ecr"].notna().sum() == 0:
        return rows
    for label in test_labels:
        test_all = df[(df["season"] == label - horizon) & df[TARGET].notna()]
        if test_all.empty:
            continue
        for cutoff in cutoff_grid:
            flag = f"eligible_next__g{cutoff}"
            elig = test_all[test_all[flag] == 1] if flag in test_all else test_all
            covered = elig[elig["market_ecr"].notna()]
            if len(covered) < 2:
                continue
            pred = -covered["market_ecr"].to_numpy(dtype=float)  # lower rank = better
            rows.append({**base_key, "model_type": "benchmark", "model": "market_ecr",
                         "cutoff_games": int(cutoff), "test_season": int(label),
                         "n_eligible": int(len(elig)), "n_market_ranked": int(len(covered)),
                         "coverage": float(len(covered) / len(elig)),
                         **_eval_row(covered, pred, k_tiers)})
    return rows


def _benchmark_comparison(df, eras, block_columns, test_labels, horizon, window, combine,
                          candidates, g_star, k_tiers, earliest, seed, base_key):
    """Head-to-head: each era model vs the market on the IDENTICAL eligible∩ranked rows (§7.4).

    Fixed headline config (longest window, default combiner, chosen cutoff g*). Each fold is
    restricted to the rows the market ranked; both the model and −ECR are scored on exactly those
    rows so the 'do we beat the market?' comparison is apples-to-apples.
    """
    if "market_ecr" not in df or df["market_ecr"].notna().sum() == 0:
        return []
    n_min, n_max = train_window_bounds(window, test_labels, horizon=horizon,
                                       earliest_season=earliest)
    train = df[(df["season"] >= n_min) & (df["season"] <= n_max) & df[TARGET].notna()]
    if train.empty:
        return []
    p_key = f"precision_at_{k_tiers[0]}"
    flag = f"eligible_next__g{g_star}"
    fitted = {m: EraEnsemble(m, eras, block_columns, combine=combine, target_col=TARGET,
                             seed=seed).fit(train) for m in candidates}
    rows = []
    for label in test_labels:
        test_all = df[(df["season"] == label - horizon) & df[TARGET].notna()]
        elig = test_all[test_all[flag] == 1] if flag in test_all else test_all
        covered = elig[elig["market_ecr"].notna()]
        if len(covered) < 3:
            continue
        common = {**base_key, "test_season": int(label), "window_years": window,
                  "combine": combine, "cutoff_games": g_star, "n": int(len(covered))}
        b = ranking_metrics(covered[TARGET], -covered["market_ecr"].to_numpy(dtype=float),
                            k_tiers=k_tiers)
        rows.append({**common, "model": "market_ecr", "spearman": b["spearman"],
                     "weighted_tau": b["weighted_tau"], p_key: b[p_key]})
        for model, ens in fitted.items():
            m = ranking_metrics(covered[TARGET], ens.predict(covered), k_tiers=k_tiers)
            rows.append({**common, "model": model, "spearman": m["spearman"],
                         "weighted_tau": m["weighted_tau"], p_key: m[p_key]})
    return rows


def _availability_over_folds(train, df, eras, block_columns, test_labels, horizon, g_star,
                             seed, key):
    """Fit the N+1 games count model on the window; score each held-out season (§7.4)."""
    cols = _all_feature_columns(block_columns, df.columns)
    tr = train[train[GAMES_TARGET].notna()]
    rows = []
    if len(tr) < 30:
        return rows
    est = make_availability_estimator(seed=seed)
    est.fit(tr[cols], tr[GAMES_TARGET])
    for label in test_labels:
        feat_season = label - horizon
        test = df[(df["season"] == feat_season) & df[GAMES_TARGET].notna()]
        if len(test) < 5:
            continue
        pred = est.predict(test[cols])
        base = test["games"].to_numpy(dtype=float)  # prior-season games baseline
        m = availability_metrics(test[GAMES_TARGET], pred, cutoff=g_star)
        b = availability_metrics(test[GAMES_TARGET], base, cutoff=g_star)
        rows.append({**key, "test_season": int(label), "model": "gbm_poisson", **m})
        rows.append({**key, "test_season": int(label), "model": "baseline_prior_games", **b})
    return rows


def _ngs_ablation(train, df, eras, block_columns, test_feat_seasons, cutoff_grid, k_tiers,
                  g_star, seed, key):
    """Fit the ngs era model with vs without the ngs_efficiency block; report the delta (§7.3).

    Scored on the high-coverage board (the chosen cutoff g*) over the ngs-era test rows. A
    positive ``delta_*`` means NGS earns its place; the decision rule keeps the block iff it
    improves top-k out-of-fold.
    """
    import numpy as np

    ngs_era = next((e for e in eras if e.name == "ngs"), None)
    if ngs_era is None:
        return []
    ngs_train = train[train["season"].map(lambda s: assign_era(int(s), eras)) == "ngs"]
    if len(ngs_train) < 30:
        return []

    full_cols = _dedupe([c for c in feature_columns_for_era(ngs_era, block_columns)
                         if c in df.columns])
    drop = set(block_columns.get("ngs_efficiency", []))
    reduced_cols = [c for c in full_cols if c not in drop]

    test = df[df["season"].isin(test_feat_seasons) & df[TARGET].notna()]
    flag = f"eligible_next__g{g_star}"
    test_elig = test[test[flag] == 1] if flag in test else test
    if len(test_elig) < 10:
        return []

    from ..models.zoo import make_estimator
    rows = []
    for variant, cols in [("with_ngs", full_cols), ("without_ngs", reduced_cols)]:
        est = make_estimator("lightgbm", seed=seed)
        est.fit(ngs_train[cols], ngs_train[TARGET])
        per_season = []
        for _, g in test_elig.groupby("season"):
            r = ranking_metrics(g[TARGET], est.predict(g[cols]), k_tiers=k_tiers)
            per_season.append(r)
        agg = {m: float(np.nanmean([r[m] for r in per_season]))
               for m in per_season[0] if m != "n"}
        rows.append({**key, "variant": variant, "n_cols": len(cols), **agg})
    # decision: keep ngs block iff it improves the top tier out-of-fold
    if len(rows) == 2:
        with_, without = rows[0], rows[1]
        key_metric = f"precision_at_{k_tiers[0]}"
        delta = with_[key_metric] - without[key_metric]
        for r in rows:
            r["keep_ngs_block"] = bool(delta > 0)
            r[f"delta_{key_metric}"] = delta
    return rows


def _dedupe(seq):
    out, seen = [], set()
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _aggregate_ranking(rank_df):
    """Mean ± sd across folds for each (model, window, combine, cutoff)."""
    import pandas as pd

    if rank_df.empty:
        return pd.DataFrame()
    keys = ["sport", "position", "model_type", "model", "window_years", "combine",
            "cutoff_games"]
    metric_cols = [c for c in ["mae", "rmse", "r2", "spearman", "kendall", "weighted_tau", *[
        c for c in rank_df.columns if c.startswith(("ndcg_at_", "precision_at_"))]]
        if c in rank_df.columns]
    g = rank_df.groupby(keys)[metric_cols]
    agg = g.mean().add_suffix("_mean").join(g.std().add_suffix("_sd"))
    agg["n_folds"] = g.size()
    return agg.reset_index()


def _recency_table(agg):
    """Recency-bias read: headline Spearman by window at the chosen cutoff, per model."""
    import pandas as pd

    if agg.empty:
        return pd.DataFrame()
    sub = agg[(agg["model_type"] == "era_ensemble")]
    keep = ["model", "combine", "cutoff_games", "window_years",
            "spearman_mean", "spearman_sd", "precision_at_12_mean", "mae_mean"]
    return sub[keep].sort_values(["model", "combine", "cutoff_games", "window_years"])


def _write_summary(results, base_key, test_labels, windows, g_star, out_dir, stem):
    import json

    agg = results["ranking_aggregate"]
    headline = {}
    if not agg.empty:
        at_star = agg[(agg["cutoff_games"] == g_star)]
        best = at_star.sort_values("spearman_mean", ascending=False).head(1)
        if len(best):
            b = best.iloc[0]
            headline = {"best_model": b["model"], "best_combine": b["combine"],
                        "window_years": int(b["window_years"]),
                        "spearman_mean": float(b["spearman_mean"]),
                        "precision_at_12_mean": float(b["precision_at_12_mean"]),
                        "mae_mean": float(b["mae_mean"])}
            baselines = at_star[at_star["model_type"] == "baseline"]
            if len(baselines):
                bb = baselines.sort_values("spearman_mean", ascending=False).iloc[0]
                headline["best_baseline"] = bb["model"]
                headline["best_baseline_spearman_mean"] = float(bb["spearman_mean"])
    summary = {**base_key, "test_label_seasons": [int(s) for s in test_labels],
               "windows": list(windows), "chosen_cutoff_games": g_star,
               "headline": headline}
    with open(out_dir / f"experiment_{stem}_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
