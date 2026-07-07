"""Stage 9 — reporting (PROJECT_PLAN §3 ``[9] report``, §8).

Reads the Stage-8 result tables from ``reports/results/`` and renders a single human-readable
Markdown report (``reports/REPORT_<sport>_<position>.md``) plus summary figures. Pure assembly —
no modeling — so the report always reflects the latest experiment run.
"""

from __future__ import annotations


def _git_provenance():
    """Return 'branch @ shortsha[-dirty]' for the working tree, or None if unavailable."""
    import subprocess

    def _git(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    try:
        branch = _git("rev-parse", "--abbrev-ref", "HEAD")
        sha = _git("rev-parse", "--short", "HEAD")
        dirty = "-dirty" if _git("status", "--porcelain") else ""
        return f"{branch} @ {sha}{dirty}"
    except Exception:
        return None


def _fmt(x, nd=3):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def _read(results_dir, stem, name):
    import pandas as pd
    path = results_dir / f"experiment_{stem}_{name}.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        # A position with no rows for this table (e.g. K has no NGS block, so
        # ngs_ablation.csv is written empty) — same as the file not existing.
        return pd.DataFrame()


def build_report(config, *, write: bool = True):
    """Assemble the Markdown results report + figures; return the report text."""
    import json
    from datetime import date

    from ..utils.io import REPORTS_DIR, ensure_dir

    sport = config.get("experiment.sport", "sport")
    position = config.require("experiment.position")
    g_star = int(config.get("eligibility.chosen_games_played", 4))
    stem = config.stem()
    results_dir = REPORTS_DIR / "results"

    agg = _read(results_dir, stem, "ranking_aggregate")
    ablation = _read(results_dir, stem, "ngs_ablation")
    avail = _read(results_dir, stem, "availability")
    bench = _read(results_dir, stem, "benchmark")
    cmp = _read(results_dir, stem, "benchmark_comparison")
    cost = _read(results_dir, stem, "cost")
    summary_path = results_dir / f"experiment_{stem}_summary.json"
    summary = json.load(open(summary_path)) if summary_path.exists() else {}

    version = config.get("experiment.version", None)
    title_version = f" {position} {version}" if version else f" {position}"
    lines = [f"# Position Predictor — Results: {sport}{title_version}", ""]

    provenance = " · ".join(p for p in [
        f"**{position} {version}**" if version else None,
        _git_provenance(),
    ] if p)
    if provenance:
        lines += [f"_{provenance}_", ""]

    test_seasons = summary.get("test_label_seasons", [])
    lines += [f"_Generated {date.today().isoformat()}._ "
              f"Test seasons **{test_seasons}**, eligibility cutoff **g\\* = {g_star} games**. "
              "Ranking is computed within each test season; metrics are mean ± sd across the "
              "season folds.", ""]

    lines += _headline_section(agg, cmp, g_star, sport, position)
    lines += _window_section(agg, g_star)
    lines += _benchmark_section(bench, cmp, g_star, position)
    lines += _ablation_section(ablation)
    lines += _availability_section(avail)
    lines += _sensitivity_section(agg)
    lines += _data_volume_section(summary)
    lines += _compute_section(cost, agg, g_star, summary)
    lines += ["", "## Figures", "",
              f"![Spearman by window](figures/report_{stem}_windows.png)", "",
              f"![Model vs market](figures/report_{stem}_vs_market.png)", ""]
    text = "\n".join(lines)

    if not write:
        return text
    ensure_dir(REPORTS_DIR)
    (REPORTS_DIR / f"REPORT_{stem}.md").write_text(text)
    _plot_windows(agg, g_star, ensure_dir(REPORTS_DIR / "figures"), stem)
    _plot_vs_market(cmp, ensure_dir(REPORTS_DIR / "figures"), stem)
    if version:
        _write_version_snapshot(stem, version, text, results_dir, summary_path)
    return text


def _data_volume_section(summary):
    dv = summary.get("data_volume") or {}
    if not dv:
        return []
    out = ["", "## Data volume", "",
           f"- **Feature rows:** {dv.get('n_feature_rows', '—')} "
           f"({dv.get('n_labeled_rows', '—')} labeled with a next-season target)",
           f"- **Features:** {dv.get('n_features', '—')} columns",
           f"- **Seasons:** {dv.get('season_min', '—')}–{dv.get('season_max', '—')} "
           f"({dv.get('n_seasons', '—')} seasons)"]
    per_era = dv.get("per_era_rows") or {}
    if per_era:
        out.append("- **Labeled rows per era:** "
                   + ", ".join(f"{k} {v}" for k, v in per_era.items()))
    return out


def _compute_section(cost, agg, g_star, summary):
    if cost is None or cost.empty:
        return []
    comp = summary.get("compute") or {}
    out = ["", "## Compute & efficiency", ""]
    wall = comp.get("wall_seconds")
    total_fit = comp.get("total_fit_seconds")
    if wall is not None:
        out.append(f"- **Experiment wall-clock:** {wall:.1f}s "
                   f"(total model fit time {total_fit:.1f}s across the grid)"
                   if total_fit is not None else f"- **Experiment wall-clock:** {wall:.1f}s")
    # per-model fit cost at the headline window, joined to its ranking quality
    hl = summary.get("headline") or {}
    window = hl.get("window_years")
    sub = cost[cost["window_years"] == window] if window else cost
    sub = sub[sub["model_type"] == "era_ensemble"]
    if not sub.empty and not agg.empty:
        agg_w = agg[(agg["cutoff_games"] == g_star) & (agg["window_years"] == window)]
        merged = sub.merge(
            agg_w[["model", "combine", "spearman_mean", "precision_at_12_mean"]],
            on=["model", "combine"], how="left")
        merged = merged.sort_values("spearman_mean", ascending=False)
        out += ["",
                f"Per-model cost vs ranking quality at the headline window "
                f"({int(window)}-yr, g\\*={g_star}). **Efficiency** = Spearman per fit-second:",
                "",
                "| model | combine | fit (s) | train rows | features | Spearman | P@12 | "
                "efficiency |",
                "|---|---|---|---|---|---|---|---|"]
        for _, r in merged.iterrows():
            fs = r["fit_seconds"]
            eff = (r["spearman_mean"] / fs) if fs and fs > 0 else None
            out.append(f"| {r['model']} | {r['combine']} | {fs:.2f} | "
                       f"{int(r['n_train_rows'])} | {int(r['n_features'])} | "
                       f"{_fmt(r['spearman_mean'])} | {_fmt(r['precision_at_12_mean'], 2)} | "
                       f"{_fmt(eff, 2)} |")
    return out


def _write_version_snapshot(stem, version, report_text, results_dir, summary_path):
    """Archive a committed, immutable snapshot of this version under reports/versions/."""
    import shutil

    from ..utils.io import REPORTS_DIR, ensure_dir

    snap = ensure_dir(REPORTS_DIR / "versions" / stem / str(version))
    (snap / f"REPORT_{stem}_{version}.md").write_text(report_text)
    if summary_path.exists():
        shutil.copyfile(summary_path, snap / "summary.json")
    for name in ("cost", "benchmark_comparison", "recency"):
        src = results_dir / f"experiment_{stem}_{name}.csv"
        if src.exists():
            shutil.copyfile(src, snap / f"{name}.csv")


def _best_at_star(agg, g_star, model_type):
    sub = agg[(agg["cutoff_games"] == g_star) & (agg["model_type"] == model_type)]
    return sub.sort_values("spearman_mean", ascending=False).head(1)


def _headline_section(agg, cmp, g_star, sport, position):
    out = ["## Headline", ""]
    if agg.empty:
        return out + ["_No results found — run `make experiment` first._", ""]
    best = _best_at_star(agg, g_star, "era_ensemble")
    base = _best_at_star(agg, g_star, "baseline")
    if len(best):
        b = best.iloc[0]
        out.append(f"- **Best model:** `{b['model']}` ({b['combine']}, {int(b['window_years'])}-yr "
                   f"window) — Spearman **{_fmt(b['spearman_mean'])} ± {_fmt(b['spearman_sd'])}**, "
                   f"Precision@12 {_fmt(b['precision_at_12_mean'],2)}, MAE {_fmt(b['mae_mean'],2)} "
                   "PPG (full eligible universe).")
    if len(base):
        bb = base.iloc[0]
        out.append(f"- **Best baseline:** `{bb['model']}` — Spearman {_fmt(bb['spearman_mean'])} "
                   "(the must-beat floor).")
    if not cmp.empty:
        m = cmp.groupby("model")["spearman"].mean()
        mk = m.get("market_ecr", float("nan"))
        best_model = m.drop(labels=["market_ecr"], errors="ignore").sort_values(
            ascending=False)
        if len(best_model):
            mn = best_model.index[0]
            verdict = "does **not** beat" if best_model.iloc[0] < mk else "**beats**"
            out.append(f"- **Market head-to-head** (FantasyPros preseason ECR, scored on the "
                       f"identical rows the market ranks): market Spearman {_fmt(mk)} vs our best "
                       f"`{mn}` {_fmt(best_model.iloc[0])} — the model {verdict} the market on "
                       "overall rank.")
            # Tiers are position-configured (precision_at_<k> columns); the two smallest k are
            # the headline tiers (e.g. RB1/RB2, or QB1/QB2 at k=6/12), labelled by position.
            prec = _precision_cols(cmp)
            for i, (k, tier) in enumerate(prec[:2], start=1):
                label = f"Precision@{k} (tier-{i} / {position}{i})"
                pt = cmp.groupby("model")[tier].mean()
                pt_best = pt.drop(labels=["market_ecr"], errors="ignore").sort_values(
                    ascending=False)
                if len(pt_best):
                    v = "beats" if pt_best.iloc[0] > pt.get("market_ecr") else "trails"
                    out.append(f"  On **{label}**: market {_fmt(pt.get('market_ecr'),2)} vs best "
                               f"`{pt_best.index[0]}` {_fmt(pt_best.iloc[0],2)} — model {v} market.")
            if "weighted_tau" in cmp.columns:
                wt = cmp.groupby("model")["weighted_tau"].mean()
                wt_best = wt.drop(labels=["market_ecr"], errors="ignore").sort_values(
                    ascending=False)
                if len(wt_best):
                    wv = "does **not** beat" if wt_best.iloc[0] < wt.get("market_ecr") \
                        else "**beats**"
                    out.append(f"  On the **top-weighted** rank score (Weighted τ — errors near "
                               f"#1 count most): market {_fmt(wt.get('market_ecr'))} vs "
                               f"`{wt_best.index[0]}` {_fmt(wt_best.iloc[0])} — the model {wv} "
                               "the market where it matters most.")
    out.append("")
    return out


def _window_section(agg, g_star):
    out = ["## Recency: how much history helps (§6.2)", ""]
    if agg.empty:
        return out + ["_n/a_", ""]
    sub = agg[(agg["cutoff_games"] == g_star) & (agg["model_type"] == "era_ensemble")]
    if sub.empty:
        return out + ["_n/a_", ""]
    piv = sub.pivot_table(index=["model", "combine"], columns="window_years",
                          values="spearman_mean")
    out.append("Spearman by training-window length (years), at g\\*:")
    out.append("")
    out.append("| model | combine | " + " | ".join(f"{int(w)}yr" for w in piv.columns) + " |")
    out.append("|---|---|" + "|".join("---" for _ in piv.columns) + "|")
    for (model, combine), row in piv.iterrows():
        out.append(f"| {model} | {combine} | "
                   + " | ".join(_fmt(row[w]) for w in piv.columns) + " |")
    out.append("")
    out.append("> 20 yr ≈ 30 yr is expected — the box-score era is data-capped at 1999.")
    out.append("")
    return out


def _precision_cols(frame):
    """Sorted [(k, 'precision_at_<k>')] for the precision tiers present in a result frame."""
    return sorted((int(c.rsplit("_", 1)[1]), c) for c in frame.columns
                  if c.startswith("precision_at_") and c.rsplit("_", 1)[1].isdigit())


def _benchmark_section(bench, cmp, g_star, position):
    out = ["## Market benchmark (§7.4)", ""]
    if bench.empty:
        return out + ["_No market benchmark available (run `make benchmark`)._", ""]
    sub = bench[bench["cutoff_games"] == g_star]
    tier1_k, tier1_col = _precision_cols(bench)[0]  # smallest tier = tier-1 board
    out.append("Preseason ECR coverage of the eligible universe and the market's own ranking "
               "quality, per test season:")
    out.append("")
    out.append(f"| season | eligible | ranked | coverage | market Spearman | market P@{tier1_k} |")
    out.append("|---|---|---|---|---|---|")
    for _, r in sub.sort_values("test_season").iterrows():
        out.append(f"| {int(r['test_season'])} | {int(r['n_eligible'])} | "
                   f"{int(r['n_market_ranked'])} | {_fmt(r['coverage'],2)} | "
                   f"{_fmt(r['spearman'])} | {_fmt(r[tier1_col],2)} |")
    out.append("")
    if not cmp.empty:
        out.append("**Head-to-head on the identical ranked rows** (mean across folds). "
                   "`weighted_tau` is the **top-weighted** rank score — errors near #1 count most:")
        out.append("")
        prec = _precision_cols(cmp)
        metric_cols = [c for c in ["spearman", "weighted_tau"] if c in cmp.columns] \
            + [c for _, c in prec]
        m = cmp.groupby("model")[metric_cols].mean().sort_values(
            "weighted_tau" if "weighted_tau" in metric_cols else "spearman", ascending=False)
        head = {"spearman": "Spearman", "weighted_tau": "Weighted τ (top)"}
        for i, (k, c) in enumerate(prec, start=1):
            head[c] = f"Precision@{k} ({position}{i})"
        out.append("| model | " + " | ".join(head[c] for c in metric_cols) + " |")
        out.append("|---|" + "|".join("---" for _ in metric_cols) + "|")
        for model, r in m.iterrows():
            tag = " _(market)_" if model == "market_ecr" else ""
            out.append(f"| {model}{tag} | "
                       + " | ".join(_fmt(r[c], 2 if c.startswith("precision_at_") else 3)
                                    for c in metric_cols) + " |")
        out.append("")
    return out


def _ablation_section(ablation):
    out = ["## NGS-block ablation (§7.3)", ""]
    if ablation.empty:
        return out + ["_n/a_", ""]
    keep = bool(ablation["keep_ngs_block"].iloc[0]) if "keep_ngs_block" in ablation else False
    one = ablation[ablation["window_years"] == ablation["window_years"].max()]
    with_ = one[one["variant"] == "with_ngs"]
    without = one[one["variant"] == "without_ngs"]
    if len(with_) and len(without):
        w, wo = with_.iloc[0], without.iloc[0]
        out.append(f"`ngs` era model, longest window — **with** NGS: Spearman {_fmt(w['spearman'])}, "
                   f"P@12 {_fmt(w['precision_at_12'],2)}; **without**: Spearman "
                   f"{_fmt(wo['spearman'])}, P@12 {_fmt(wo['precision_at_12'],2)}.")
    out.append(f"- **Decision:** {'keep' if keep else 'drop'} the `ngs_efficiency` block — "
               f"{'it improves' if keep else 'no top-12 gain out-of-fold, so the ngs era collapses to the snaps schema'}"
               " (coverage flags retained either way).")
    out.append("")
    return out


def _availability_section(avail):
    out = ["## Availability model (§7.4)", ""]
    if avail.empty:
        return out + ["_n/a_", ""]
    m = avail.groupby("model")[["games_mae", "clears_auc"]].mean()
    out.append("Predicting N+1 games played (gates projected eligibility / injury risk):")
    out.append("")
    out.append("| model | games MAE | clears-cutoff AUC |")
    out.append("|---|---|---|")
    for model, r in m.iterrows():
        out.append(f"| {model} | {_fmt(r['games_mae'],2)} | {_fmt(r['clears_auc'])} |")
    out.append("")
    return out


def _sensitivity_section(agg):
    out = ["## Eligibility-cutoff sensitivity (§6.4)", ""]
    if agg.empty:
        return out + ["_n/a_", ""]
    best_model = None
    e = agg[agg["model_type"] == "era_ensemble"]
    if e.empty:
        return out + ["_n/a_", ""]
    best_model = e.sort_values("spearman_mean", ascending=False).iloc[0]["model"]
    sub = e[(e["model"] == best_model)].groupby("cutoff_games")["spearman_mean"].mean()
    out.append(f"Best model (`{best_model}`) Spearman across the candidate games cutoffs "
               "(robustness — the headline does not hinge on g\\*):")
    out.append("")
    out.append("| cutoff (games) | " + " | ".join(str(int(c)) for c in sub.index) + " |")
    out.append("|---|" + "|".join("---" for _ in sub.index) + "|")
    out.append("| Spearman | " + " | ".join(_fmt(v) for v in sub.values) + " |")
    out.append("")
    return out


def _plot_windows(agg, g_star, fig_dir, stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if agg.empty:
        return
    sub = agg[(agg["cutoff_games"] == g_star) & (agg["model_type"] == "era_ensemble") &
              (agg["combine"] == agg["combine"].iloc[0])]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model, g in sub.groupby("model"):
        g = g.sort_values("window_years")
        ax.plot(g["window_years"], g["spearman_mean"], marker="o", label=model)
    base = agg[(agg["cutoff_games"] == g_star) & (agg["model_type"] == "baseline")]
    if len(base):
        ax.axhline(base["spearman_mean"].max(), ls="--", color="grey",
                   label="best baseline")
    ax.set(xlabel="training window (years)", ylabel="Spearman ρ (mean across folds)",
           title=f"Ranking quality by history window (g*={g_star})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / f"report_{stem}_windows.png", dpi=120)
    plt.close(fig)


def _plot_vs_market(cmp, fig_dir, stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if cmp.empty:
        return
    m = cmp.groupby("model")[["spearman", "precision_at_12"]].mean().sort_values("spearman")
    colors = ["tab:orange" if i == "market_ecr" else "tab:blue" for i in m.index]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.barh(m.index, m["spearman"], color=colors)
    ax1.set(xlabel="Spearman ρ", title="Model vs market — overall rank (covered rows)")
    ax2.barh(m.index, m["precision_at_12"], color=colors)
    ax2.set(xlabel="Precision@12", title="Model vs market — top-12 hit rate")
    fig.tight_layout()
    fig.savefig(fig_dir / f"report_{stem}_vs_market.png", dpi=120)
    plt.close(fig)
