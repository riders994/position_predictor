"""Cross-version progress report.

Scans the committed per-version snapshots under ``reports/versions/{stem}/`` and
assembles a single ``PROGRESS_{stem}.md`` that tracks, version over version, the
ranking quality of the best model alongside what it cost to get there — compute
(fit / wall-clock seconds) and data volume (rows, features). This is how we judge
whether a new version's gains are worth their added compute / data.
"""
from __future__ import annotations

import json
import re

import pandas as pd

from .report import _fmt


def _version_key(v: str):
    """Sort 'v10' after 'v2'; fall back to lexical for non-vN labels."""
    m = re.match(r"v?(\d+)", str(v))
    return (0, int(m.group(1))) if m else (1, str(v))


def _best_market_row(snap_dir):
    """Aggregate benchmark_comparison (mean across folds); return (best_model, market) dicts."""
    path = snap_dir / "benchmark_comparison.csv"
    if not path.exists():
        return None, None
    df = pd.read_csv(path)
    if df.empty:
        return None, None
    metrics = ["spearman", "weighted_tau", "precision_at_12", "precision_at_24"]
    agg = df.groupby("model")[metrics].mean().reset_index()
    market = agg[agg["model"] == "market_ecr"]
    models = agg[agg["model"] != "market_ecr"].sort_values("spearman", ascending=False)
    best = models.iloc[0].to_dict() if not models.empty else None
    mkt = market.iloc[0].to_dict() if not market.empty else None
    return best, mkt


def _collect(stem, versions_dir):
    rows = []
    for snap_dir in sorted(versions_dir.iterdir(), key=lambda p: _version_key(p.name)):
        if not snap_dir.is_dir():
            continue
        summ_path = snap_dir / "summary.json"
        summary = json.load(open(summ_path)) if summ_path.exists() else {}
        best, mkt = _best_market_row(snap_dir)
        rows.append({
            "version": snap_dir.name,
            "summary": summary,
            "best": best,
            "market": mkt,
        })
    return rows


def build_progress(config, *, write: bool = True):
    """Assemble the cross-version progress report; return its Markdown text (or None)."""
    from ..utils.io import REPORTS_DIR

    sport = config.get("experiment.sport", "sport")
    position = config.require("experiment.position")
    stem = f"{sport}_{position}".lower()
    versions_dir = REPORTS_DIR / "versions" / stem
    if not versions_dir.exists():
        return None
    rows = _collect(stem, versions_dir)
    if not rows:
        return None

    lines = [f"# Position Predictor — Progress across versions: {sport} {position}", "",
             "Best model per version (head-to-head vs market on covered rows) against the "
             "compute and data volume it took. Use this to judge whether a version's ranking "
             "gains justified its added cost.", ""]

    # ---- ranking quality vs market ----
    lines += ["## Ranking quality (best model vs market)", "",
              "| version | best model | Spearman | Weighted τ | P@12 | P@24 | "
              "Δ Spearman vs market |", "|---|---|---|---|---|---|---|"]
    prev_spear = None
    for r in rows:
        b, m = r["best"], r["market"]
        if not b:
            continue
        model = f"{b['model']}"
        d_mkt = (b["spearman"] - m["spearman"]) if m else None
        delta = f" ({b['spearman'] - prev_spear:+.3f} vs prev)" if prev_spear is not None else ""
        lines.append(
            f"| {r['version']} | {model} | {_fmt(b['spearman'])}{delta} | "
            f"{_fmt(b['weighted_tau'])} | {_fmt(b['precision_at_12'], 2)} | "
            f"{_fmt(b['precision_at_24'], 2)} | {_fmt(d_mkt)} |")
        prev_spear = b["spearman"]

    # ---- compute & data volume ----
    lines += ["", "## Cost: compute & data volume", "",
              "| version | wall (s) | total fit (s) | features | labeled rows | "
              "Spearman / fit-s |", "|---|---|---|---|---|---|"]
    for r in rows:
        s, b = r["summary"], r["best"]
        comp = s.get("compute") or {}
        dv = s.get("data_volume") or {}
        wall = comp.get("wall_seconds")
        total_fit = comp.get("total_fit_seconds")
        eff = None
        if b and comp.get("mean_fit_seconds_per_model"):
            mf = comp["mean_fit_seconds_per_model"]
            eff = b["spearman"] / mf if mf else None
        lines.append(
            f"| {r['version']} | {_fmt(wall, 1) if wall is not None else '—'} | "
            f"{_fmt(total_fit, 1) if total_fit is not None else '—'} | "
            f"{dv.get('n_features', '—')} | {dv.get('n_labeled_rows', '—')} | "
            f"{_fmt(eff, 2)} |")

    lines += ["", "_Snapshots live in `reports/versions/<stem>/<version>/` "
              "(committed; the big regenerable CSVs stay gitignored)._", ""]
    text = "\n".join(lines)

    if not write:
        return text
    (versions_dir / f"PROGRESS_{stem}.md").write_text(text)
    return text
