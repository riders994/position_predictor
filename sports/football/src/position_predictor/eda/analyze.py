"""Stage 6 — exploratory data analysis (PROJECT_PLAN §3 ``[6] eda``).

Three reads over the Stage-4 processed modeling matrix, each a pure function returning a tidy
DataFrame so the notebooks (``notebooks/football/rb/01,02``) and the headless ``run_eda.py``
script share one implementation (no copy-paste logic):

1. **Coverage** — how the *ranking universe* and the *feature schema* change over time. Makes
   the nflverse data caps (snaps 2012+, NGS 2016+) and the era partition (PROJECT_PLAN §6.3)
   visible empirically, plus the trailing-season **target censoring** (the latest feature
   season has no observed N+1 yet) and the NGS-by-tier sparsity that motivates the Stage-8
   ablation (§7.3).
2. **Distributions** — shape of the target and key features (skew, zero-inflation, spread) so
   modeling choices (transforms, tree vs linear, mean-reversion) are grounded in the data.
3. **Target stability** — how well season *N* predicts season *N+1* (rank Spearman, the
   persistence-baseline error, regression-to-the-mean). This is the must-beat signal the
   Stage-8 models are built to exploit and the empirical basis of the §7.1 baselines.

All functions take DataFrames and lazily import ``pandas``/``numpy``; the orchestrator
:func:`run_eda` does the I/O (CSV tables + figures under ``reports/``), mirroring Stage 5.
"""

from __future__ import annotations

# ----------------------------------------------------------------------------- coverage


def season_universe(season_df, eras=None):
    """Per-season size of the RB universe + next-season target availability.

    For each feature season *N*: row count, era (if ``eras`` supplied), the breakdown of
    ``status_next`` (active / injured_out / retired), the **target coverage** (share with an
    observed N+1 PPG), and mean games/PPG. The most recent season shows the structural
    censoring — its N+1 has not happened (or is unpublished), so target coverage drops.
    """
    import numpy as np
    import pandas as pd

    from ..eras import assign_era

    df = season_df
    rows = []
    for season, g in df.groupby("season"):
        n = len(g)
        status = g.get("status_next")
        vc = status.value_counts() if status is not None else pd.Series(dtype=int)
        n_target = int(g["target_ppg_next"].notna().sum()) if "target_ppg_next" in g else 0
        rows.append({
            "season": int(season),
            "era": assign_era(int(season), eras) if eras else None,
            "n_player_seasons": n,
            "n_active_next": int(vc.get("active", 0)),
            "n_injured_next": int(vc.get("injured_out", 0)),
            "n_retired_next": int(vc.get("retired", 0)),
            "target_coverage": float(n_target / n) if n else np.nan,
            "mean_games": float(g["games"].mean()) if "games" in g else np.nan,
            "mean_ppg": float(g["ppg"].mean()) if "ppg" in g else np.nan,
        })
    return pd.DataFrame(rows).sort_values("season").reset_index(drop=True)


def block_coverage(df, block_columns, eras):
    """Mean non-null fraction of each feature **block**, per era.

    Confirms the nested-schema design (PROJECT_PLAN §6.3): era-gated blocks (``snap_usage``,
    ``ngs_efficiency``) are ~empty before their era and populated within it, while the
    always-on blocks stay covered throughout. ``coverage`` averages the per-column non-null
    rate over the block's columns within each era's rows.
    """
    import numpy as np
    import pandas as pd

    from ..eras import assign_era

    era_of = df["season"].map(lambda s: assign_era(int(s), eras))
    rows = []
    for era in eras:
        mask = era_of == era.name
        sub = df[mask]
        for block, cols in block_columns.items():
            present = [c for c in cols if c in sub.columns]
            if not present or sub.empty:
                cov = np.nan
            else:
                cov = float(sub[present].notna().mean().mean())
            rows.append({
                "era": era.name,
                "block": block,
                "in_era_schema": block in era.feature_blocks,
                "n_cols": len(present),
                "coverage": cov,
            })
    return pd.DataFrame(rows)


def column_coverage_by_season(df, columns):
    """Per-season non-null fraction for the given (typically era-gated) columns.

    Long/tidy ``[season, column, coverage, n]`` — the raw material for the era-boundary
    coverage figure (e.g. ``snap_share`` jumps at 2012, NGS columns at 2016).
    """
    import pandas as pd

    rows = []
    for season, g in df.groupby("season"):
        n = len(g)
        for c in columns:
            if c not in g.columns:
                continue
            rows.append({
                "season": int(season),
                "column": c,
                "coverage": float(g[c].notna().mean()),
                "n": n,
            })
    return pd.DataFrame(rows)


def ngs_coverage_by_tier(df, *, rank_col="finish_ppr_rank",
                         flag_cols=("has_ngs_rush", "has_ngs_rec"),
                         tier_edges=(12, 24, 36), min_season=2016):
    """NGS coverage by within-season fantasy-rank tier (NGS era only).

    Reproduces the §7.3 finding that NGS missingness is *signal*: coverage is high among the
    top backs and collapses for the fringe — and a top back lacking NGS *rushing* is usually a
    receiving-profile RB (NGS rushing has an attempt threshold). Tiers are ``[1..12]``,
    ``[13..24]``, ``[25..36]``, ``[37+]`` by season-N PPR finish.
    """
    import numpy as np
    import pandas as pd

    sub = df[df["season"] >= min_season]
    if rank_col not in sub.columns:
        return pd.DataFrame(columns=["tier", "n", *flag_cols])
    edges = [0, *tier_edges, np.inf]
    labels = [f"<= {tier_edges[0]}"] + \
             [f"{lo + 1}-{hi}" for lo, hi in zip(tier_edges, tier_edges[1:])] + \
             [f"{tier_edges[-1] + 1}+"]
    tier = pd.cut(sub[rank_col], bins=edges, labels=labels, right=True)
    rows = []
    for label in labels:
        grp = sub[tier == label]
        rec = {"tier": label, "n": int(len(grp))}
        for fc in flag_cols:
            rec[fc] = float(grp[fc].mean()) if fc in grp and len(grp) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------ distributions


def distribution_summary(df, columns):
    """Tidy distribution summary (n, mean, std, skew, quantiles, zero-share) per column.

    Surfaces the right-skew / zero-inflation of fantasy volume & scoring that argues for
    rank-based evaluation and tree models (PROJECT_PLAN §7–8).
    """
    import numpy as np
    import pandas as pd

    rows = []
    for c in columns:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty:
            continue
        rows.append({
            "column": c,
            "n": int(s.size),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "skew": float(s.skew()) if s.size > 2 else np.nan,
            "min": float(s.min()),
            "p05": float(s.quantile(0.05)),
            "p25": float(s.quantile(0.25)),
            "p50": float(s.quantile(0.50)),
            "p75": float(s.quantile(0.75)),
            "p95": float(s.quantile(0.95)),
            "max": float(s.max()),
            "pct_zero": float((s == 0).mean()),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------- target stability


def target_stability(season_df, *, ppg_col="ppg", target_col="target_ppg_next"):
    """Year-over-year predictability of PPG: rank Spearman + persistence-baseline error.

    The build already aligns each row's season-*N* ``ppg`` with its observed N+1
    ``target_ppg_next`` (defined only when the player is active next season), so each season's
    rows *are* the N→N+1 pairs. For each *N* we report the count, Spearman & Pearson of
    ``ppg`` vs ``target_ppg_next``, and the **persistence baseline** error
    (mean ``|ppg − ppg_next|`` — the §7.1 must-beat floor). A pooled ``season = -1`` row
    aggregates across all pairs.
    """
    import numpy as np
    import pandas as pd

    df = season_df[[ "season", ppg_col, target_col]].dropna(subset=[ppg_col, target_col])

    def _stats(g):
        x, y = g[ppg_col], g[target_col]
        err = (x - y).abs()
        return {
            "n_pairs": int(len(g)),
            "spearman": float(x.corr(y, method="spearman")) if len(g) > 2 else np.nan,
            "pearson": float(x.corr(y, method="pearson")) if len(g) > 2 else np.nan,
            "persistence_mae": float(err.mean()),
            "persistence_rmse": float(np.sqrt((err ** 2).mean())),
        }

    rows = []
    for season, g in df.groupby("season"):
        rows.append({"season": int(season), **_stats(g)})
    out = pd.DataFrame(rows).sort_values("season").reset_index(drop=True)
    pooled = pd.DataFrame([{"season": -1, **_stats(df)}])  # -1 = pooled across all seasons
    return pd.concat([out, pooled], ignore_index=True)


def regression_to_mean(season_df, *, ppg_col="ppg", target_col="target_ppg_next", n_buckets=5):
    """Mean reversion: bucket by season-*N* PPG, show mean season-*N+1* PPG per bucket.

    Quantifies the §7.1 ``mean_reversion`` baseline — high-PPG buckets regress *down* toward
    the population mean and low-PPG buckets *up*. ``shrink`` is the share of the bucket's
    distance-from-mean that disappears next season (1.0 = full reversion to the grand mean).
    """
    import numpy as np
    import pandas as pd

    df = season_df[[ppg_col, target_col]].dropna()
    if df.empty:
        return pd.DataFrame()
    grand_mean = float(df[target_col].mean())
    bucket = pd.qcut(df[ppg_col], q=n_buckets, labels=False, duplicates="drop")
    rows = []
    for b, g in df.groupby(bucket):
        mean_n = float(g[ppg_col].mean())
        mean_next = float(g[target_col].mean())
        denom = mean_n - grand_mean
        rows.append({
            "ppg_bucket": int(b),
            "n": int(len(g)),
            "mean_ppg_n": mean_n,
            "mean_ppg_next": mean_next,
            "grand_mean_next": grand_mean,
            "shrink": float(1 - (mean_next - grand_mean) / denom) if denom else np.nan,
        })
    return pd.DataFrame(rows).sort_values("ppg_bucket").reset_index(drop=True)


# --------------------------------------------------------------------------- orchestrator

# Era-gated columns whose per-season coverage traces the era boundaries (snaps 2012, NGS 2016).
ERA_GATED_COLUMNS = ["snap_share", "ngs_efficiency", "ryoe_per_att", "avg_separation",
                     "rush_pct_over_expected"]
# Headline columns for the distribution table (target + core volume/efficiency/scoring).
DISTRIBUTION_COLUMNS = ["ppg", "target_ppg_next", "ppr_points", "touches", "carries",
                        "targets", "yards_per_carry", "snap_share", "age_at_season_start",
                        "games"]


def run_eda(config, *, write: bool = True):
    """Orchestrate Stage 6: compute the coverage / distribution / stability reads + artifacts.

    Reads the Stage-4 processed matrix; returns a dict of tidy tables. When ``write`` also
    persists ``reports/results/eda_*.{json,csv}`` and two ``reports/figures/eda_*.png`` figures.
    """
    import json
    from datetime import datetime, timezone

    from ..eras import load_eras
    from ..utils.io import DATA_PROCESSED, REPORTS_DIR, ensure_dir, read_parquet
    from ..utils.naming import artifact_stem

    sport = config.get("experiment.sport", "sport")
    position = config.require("experiment.position")
    eras = load_eras(config)

    stem = artifact_stem(config)
    df = read_parquet(DATA_PROCESSED / f"{stem}_features.parquet")
    block_columns = json.load(
        open(DATA_PROCESSED / f"{stem}_feature_blocks.json"))

    tables = {
        "universe": season_universe(df, eras),
        "block_coverage": block_coverage(df, block_columns, eras),
        "column_coverage": column_coverage_by_season(df, ERA_GATED_COLUMNS),
        "ngs_tier_coverage": ngs_coverage_by_tier(df),
        "distributions": distribution_summary(df, DISTRIBUTION_COLUMNS),
        "target_stability": target_stability(df),
        "regression_to_mean": regression_to_mean(df),
    }

    if not write:
        return tables

    results_dir = ensure_dir(REPORTS_DIR / "results")
    for name, tbl in tables.items():
        tbl.to_csv(results_dir / f"eda_{stem}_{name}.csv", index=False)

    pooled = tables["target_stability"]
    pooled_row = pooled[pooled["season"] == -1].iloc[0].to_dict()
    summary = {
        "sport": sport, "position": position,
        "n_player_seasons": int(len(df)),
        "seasons": [int(df["season"].min()), int(df["season"].max())],
        "n_with_target": int(df["target_ppg_next"].notna().sum()),
        "pooled_target_stability": pooled_row,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(results_dir / f"eda_{stem}_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    fig_dir = ensure_dir(REPORTS_DIR / "figures")
    _plot_coverage(tables, fig_dir, stem)
    _plot_stability(tables, fig_dir, stem)
    return tables


def _plot_coverage(tables, fig_dir, stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cov = tables["column_coverage"]
    tier = tables["ngs_tier_coverage"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    for col, g in cov.groupby("column"):
        ax1.plot(g["season"], g["coverage"], marker=".", label=col)
    ax1.axvline(2012, ls=":", color="grey")
    ax1.axvline(2016, ls=":", color="grey")
    ax1.set(xlabel="feature season N", ylabel="non-null fraction",
            title="Era-gated feature coverage by season")
    ax1.legend(fontsize=7)
    if not tier.empty:
        ax2.bar(range(len(tier)), tier["has_ngs_rush"], width=0.4, label="has_ngs_rush",
                align="edge")
        ax2.bar([i + 0.4 for i in range(len(tier))], tier["has_ngs_rec"], width=0.4,
                label="has_ngs_rec", align="edge")
        ax2.set_xticks([i + 0.4 for i in range(len(tier))])
        ax2.set_xticklabels(tier["tier"])
        ax2.set(xlabel="PPR finish tier (season N)", ylabel="coverage",
                title="NGS coverage by tier (2016+)")
        ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / f"eda_{stem}_coverage.png", dpi=120)
    plt.close(fig)


def _plot_stability(tables, fig_dir, stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    st = tables["target_stability"]
    st = st[st["season"] >= 0]
    rtm = tables["regression_to_mean"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    ax1.plot(st["season"], st["spearman"], marker="o", label="Spearman ρ")
    ax1.plot(st["season"], st["pearson"], marker="s", label="Pearson r")
    ax1.set(xlabel="feature season N", ylabel="corr(PPG_N, PPG_N+1)",
            title="Year-over-year PPG predictability", ylim=(0, 1))
    ax1.legend(fontsize=8)
    if not rtm.empty:
        ax2.plot(rtm["ppg_bucket"], rtm["mean_ppg_n"], marker="o", label="PPG season N")
        ax2.plot(rtm["ppg_bucket"], rtm["mean_ppg_next"], marker="o", label="PPG season N+1")
        ax2.axhline(rtm["grand_mean_next"].iloc[0], ls="--", color="grey",
                    label="grand mean N+1")
        ax2.set(xlabel="season-N PPG quintile", ylabel="mean PPG",
                title="Regression to the mean")
        ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / f"eda_{stem}_stability.png", dpi=120)
    plt.close(fig)
