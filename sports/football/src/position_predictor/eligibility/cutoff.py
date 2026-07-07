"""Stage 5 — derive the games / snap-share eligibility cutoff (PROJECT_PLAN §4.2).

The cutoff defines the **ranking universe**: who is "rankable" at season end. It is derived,
documented, and sensitivity-tested — never assumed. Three complementary reads:

1. **Reliability / signal stabilization (primary, games).** Low-game PPG is high-variance
   noise. Via repeated **split-half** sampling we estimate the reliability of a *k*-game PPG
   estimate = correlation between two disjoint *k*-game samples of the same player-season,
   across players. Reliability rises with *k*; the chosen cutoff ``g*`` is the smallest *k*
   whose reliability clears ``reliability_target``. (Split-half needs 2·*k* games, so the curve
   is testable up to ~8 on 16/17-game seasons — which itself argues practical cutoffs are ≤8.)
2. **Coverage vs purity trade-off (games & snaps).** For each candidate cutoff: how much real
   fantasy production is retained vs how many would-be-relevant players are excluded.
3. **Snap-share (2012+, secondary).** Coverage across the candidate snap grid + a distribution
   elbow separating usable starters from committee fringe.

Output: the reliability curve, the full candidate grid (so downstream rank metrics are reported
across cutoffs for robustness), and the chosen rule — written to ``reports/results`` with a
figure in ``reports/figures``. Logic is pure/testable; ``run_eligibility.py`` does the I/O.

Note on leakage: PPG reliability is a stable measurement property, so the derivation runs over
all available seasons; the **sensitivity grid** is what guards conclusions, and Stage 8 applies
(does not re-fit) the chosen rule within evaluation folds (§6.4).
"""

from __future__ import annotations


def split_half_reliability(weekly, k_grid, *, ppr_col="fantasy_points_ppr",
                           n_repeats=50, seed=1729, regular_season_only=True):
    """Reliability of a *k*-game PPG estimate, per *k*, via disjoint split-half sampling.

    Returns a DataFrame ``[k, reliability, n_pool]``. For each *k* we use player-seasons with
    ≥ 2·*k* games, draw two disjoint random *k*-game samples, take each sample's mean PPR, and
    correlate the two across the pool (averaged over ``n_repeats`` random splits).
    """
    import numpy as np
    import pandas as pd

    df = weekly
    if regular_season_only and "season_type" in df.columns:
        df = df[df["season_type"] == "REG"]
    games = df.groupby(["player_id", "season"])[ppr_col].apply(lambda s: s.to_numpy())

    rng = np.random.default_rng(seed)
    rows = []
    for k in k_grid:
        pool = [a for a in games if len(a) >= 2 * k]
        if len(pool) < 5:
            rows.append({"k": int(k), "reliability": np.nan, "n_pool": len(pool)})
            continue
        corrs = []
        for _ in range(n_repeats):
            a_means, b_means = [], []
            for arr in pool:
                idx = rng.permutation(len(arr))
                a_means.append(arr[idx[:k]].mean())
                b_means.append(arr[idx[k:2 * k]].mean())
            a_means, b_means = np.asarray(a_means), np.asarray(b_means)
            if a_means.std() > 0 and b_means.std() > 0:
                corrs.append(np.corrcoef(a_means, b_means)[0, 1])
        rows.append({"k": int(k),
                     "reliability": float(np.mean(corrs)) if corrs else np.nan,
                     "n_pool": len(pool)})
    return pd.DataFrame(rows)


def choose_games_cutoff(reliability_df, target):
    """Smallest ``k`` whose reliability ≥ ``target``; falls back to the most-reliable ``k``.

    Returns ``(g_star, reached)`` where ``reached`` is False if no tested ``k`` clears target.
    """

    df = reliability_df.dropna(subset=["reliability"]).sort_values("k")
    if df.empty:
        return None, False
    hit = df[df["reliability"] >= target]
    if len(hit):
        return int(hit.iloc[0]["k"]), True
    return int(df.loc[df["reliability"].idxmax(), "k"]), False


def coverage_purity(season_df, games_grid, *, value_col="ppr_points", games_col="games",
                    ppg_col="ppg"):
    """Coverage vs purity across the candidate games grid.

    For each ``G``: rows retained, share of total fantasy production retained, and the count of
    *excluded* player-seasons that nonetheless scored at/above the included group's median PPG
    (would-be-relevant players the cutoff drops — the purity cost).
    """
    import pandas as pd

    df = season_df.dropna(subset=[games_col, value_col])
    total_pts = df[value_col].sum()
    rows = []
    for g in games_grid:
        inc = df[df[games_col] >= g]
        exc = df[df[games_col] < g]
        med = inc[ppg_col].median() if len(inc) else float("nan")
        excluded_relevant = int((exc[ppg_col] >= med).sum()) if len(exc) else 0
        rows.append({
            "games_cutoff": int(g),
            "n_eligible": int(len(inc)),
            "coverage_points": float(inc[value_col].sum() / total_pts) if total_pts else float("nan"),
            "median_ppg_eligible": float(med),
            "excluded_relevant": excluded_relevant,
        })
    return pd.DataFrame(rows)


def snap_coverage(season_df, snap_grid, *, snap_col="snap_share", value_col="ppr_points"):
    """Coverage across the candidate snap-share grid (2012+ rows that have a snap share)."""
    import pandas as pd

    df = season_df.dropna(subset=[snap_col, value_col])
    total_pts = df[value_col].sum()
    rows = []
    for s in snap_grid:
        inc = df[df[snap_col] >= s]
        rows.append({
            "snap_cutoff": float(s),
            "n_eligible": int(len(inc)),
            "coverage_points": float(inc[value_col].sum() / total_pts) if total_pts else float("nan"),
        })
    return pd.DataFrame(rows)


def derive_cutoff(config, *, write: bool = True):
    """Orchestrate Stage 5: reliability + coverage + snap analyses -> chosen rule + artifacts.

    Reads the Stage-2 interim table (season aggregates) and raw weekly (per-game PPR). Returns
    a result dict; when ``write`` also persists ``reports/results/eligibility_*.{json,csv}`` and
    a ``reports/figures/eligibility_*.png`` reliability/coverage figure.
    """
    import json
    from datetime import datetime, timezone

    from ..utils.io import (DATA_INTERIM, DATA_RAW, REPORTS_DIR, ensure_dir, read_parquet)

    sport = config.get("experiment.sport", "sport")
    position = config.require("experiment.position")
    seed = int(config.get("reproducibility.random_seed", 1729))
    target = float(config.get("eligibility.reliability_target", 0.70))
    k_max = int(config.get("eligibility.reliability_k_max", 8))
    games_grid = config.get("eligibility.candidate_games_played", [4, 6, 8, 10, 12])
    snap_grid = config.get("eligibility.candidate_snap_share", [0.30, 0.40, 0.50])

    season_df = read_parquet(DATA_INTERIM / f"{config.stem()}_player_seasons.parquet")
    weekly = read_parquet(DATA_RAW / "weekly.parquet")
    weekly = weekly[weekly["player_id"].isin(set(season_df["player_id"]))]  # eligible RBs only

    reliability = split_half_reliability(weekly, list(range(1, k_max + 1)), seed=seed)
    g_star, reached = choose_games_cutoff(reliability, target)
    coverage = coverage_purity(season_df, games_grid)
    snaps = snap_coverage(season_df, snap_grid)

    result = {
        "sport": sport, "position": position,
        "method": config.get("eligibility.selection_method", "reliability_stabilization"),
        "reliability_target": target, "seed": seed,
        "chosen_games_played": g_star, "reliability_target_reached": reached,
        "reliability_at_chosen": _lookup(reliability, g_star),
        "chosen_snap_share": None,  # snaps reported as secondary; rule is games-primary for v1
        "reliability_curve": reliability.to_dict("records"),
        "coverage_grid": coverage.to_dict("records"),
        "snap_grid": snaps.to_dict("records"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    if not write:
        return result

    results_dir = ensure_dir(REPORTS_DIR / "results")
    stem = f"eligibility_{config.stem()}"
    with open(results_dir / f"{stem}.json", "w") as fh:
        json.dump(result, fh, indent=2)
    coverage.assign(metric="coverage").to_csv(results_dir / f"{stem}_coverage.csv", index=False)
    reliability.to_csv(results_dir / f"{stem}_reliability.csv", index=False)
    _plot(reliability, coverage, target, g_star, REPORTS_DIR / "figures", stem)
    return result


def _lookup(reliability_df, k):
    if k is None:
        return None
    row = reliability_df[reliability_df["k"] == k]
    return float(row["reliability"].iloc[0]) if len(row) else None


def _plot(reliability, coverage, target, g_star, fig_dir, stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..utils.io import ensure_dir
    ensure_dir(fig_dir)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(reliability["k"], reliability["reliability"], marker="o")
    ax1.axhline(target, ls="--", color="grey", label=f"target {target:g}")
    if g_star is not None:
        ax1.axvline(g_star, ls=":", color="red", label=f"chosen g*={g_star}")
    ax1.set(xlabel="games (k)", ylabel="split-half reliability of PPG",
            title="PPG reliability vs games")
    ax1.legend()
    ax2.plot(coverage["games_cutoff"], coverage["coverage_points"], marker="o")
    ax2.set(xlabel="games cutoff (G)", ylabel="share of fantasy production retained",
            title="Coverage vs cutoff")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{stem}.png", dpi=120)
    plt.close(fig)
