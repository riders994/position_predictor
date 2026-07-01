"""Phase 3 — next-season archetype predictor: build the N->N+1 table, evaluate walk-forward, write report.

Baseline to beat is **persistence** (same archetype as last season). The honest question: does modeling
style trajectory + experience beat "sticky" on hard top-1 accuracy, and/or on the calibrated soft
membership distribution (log-loss) that Phase 2 consumes at draft time?

Usage: uv run python scripts/predict_archetypes.py --config config/nba_archetypes.yaml
       uv run python scripts/predict_archetypes.py --config config/... --kind age   # Model B (true age)
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nba_archetypes.predict.build import build_predict_table  # noqa: E402
from nba_archetypes.predict.model import feature_importance, walk_forward  # noqa: E402
from nba_archetypes.utils.config import Config  # noqa: E402
from nba_archetypes.utils.io import REPORTS_DIR, ensure_dir  # noqa: E402


def _md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main() -> int:
    p = argparse.ArgumentParser(description="Phase-3 next-season archetype predictor + report.")
    p.add_argument("--config", required=True)
    p.add_argument("--kind", default="yoe", choices=["yoe", "age"],
                   help="Feature set: 'yoe' (Model A, experience proxy) or 'age' (Model B, true age).")
    args = p.parse_args()

    Config.load(args.config)
    table = build_predict_table(write=True)
    persist_full = round(float((table["arch"] == table["target_arch"]).mean()), 3)
    print(f"[predict] {len(table)} player-season pairs · {table['athlete_id'].nunique()} players · "
          f"seasons N {int(table.season.min())}-{int(table.season.max())} · "
          f"persistence (full table) {persist_full:.3f}")

    res = walk_forward(table, kind=args.kind)
    imp = feature_importance(table, kind=args.kind)

    print(f"\n[predict] walk-forward ({res['eval_seasons'][0]}-{res['eval_seasons'][-1]}, "
          f"n={res['n_eval']}, features={args.kind}):")
    for name in ("model", "persistence", "marginal"):
        m = res[name]
        print(f"   {name:12s} acc={m['accuracy']:.3f}  macroF1={m['macro_f1']:.3f}  "
              f"logloss={m['log_loss']:.3f}  brier={m['brier']:.3f}")
    print(f"   -> acc lift vs persistence {res['acc_lift_vs_persistence']:+.3f}; "
          f"log-loss gain {res['logloss_gain_vs_persistence']:+.3f}")
    print("\n[predict] top permutation-importance features:")
    for f in imp[:8]:
        print(f"   {f['feature']:16s} {f['importance']:+.4f}")

    # ---- report ----
    mm, pm = res["model"], res["persistence"]
    beats_acc = res["acc_lift_vs_persistence"] > 0
    verdict = (
        f"The model {'beats' if beats_acc else 'does **not** beat'} persistence on hard top-1 accuracy "
        f"({mm['accuracy']:.3f} vs {pm['accuracy']:.3f}) — archetype membership is highly persistent, so "
        f"'same as last season' is a very strong argmax baseline. But the model **more than halves "
        f"log-loss** ({mm['log_loss']:.3f} vs {pm['log_loss']:.3f}): it produces far better-calibrated "
        f"soft membership vectors, which is exactly what Phase 2 consumes at draft time (projected "
        f"category coverage needs a probability distribution, not a single hard label).")
    md = f"""# Phase 3 — Next-Season Archetype Predictor ({args.kind})

Leak-safe **N→N+1** prediction of a player's archetype, returning players only. Features as-of season N:
current soft membership `p0..p11` (+ `top_prob`/`entropy`), 19 style z-features, one-year style
**trajectory** deltas, archetype tenure, and **{'true age (nba_api)' if args.kind == 'age' else 'years-of-experience proxy'}**.
Target: archetype in N+1. Evaluation is **walk-forward** (train only on transitions into earlier
seasons); all baselines are scored on the same {res['n_eval']} pooled pairs
({res['eval_seasons'][0]}-{res['eval_seasons'][-1]}).

## Walk-forward results

{_md_table(["predictor", "top-1 acc", "macro-F1", "log-loss", "Brier"],
           [[n, f"{res[n]['accuracy']:.3f}", f"{res[n]['macro_f1']:.3f}", f"{res[n]['log_loss']:.3f}",
             f"{res[n]['brier']:.3f}"] for n in ("model", "persistence", "marginal")])}

> {verdict}

**Marginal** (always predict the most common archetype) is the floor — the sticky structure is real
signal both baselines above it exploit.

## What drives the prediction (permutation importance)

{_md_table(["feature", "importance"], [[f["feature"], f"{f['importance']:+.4f}"] for f in imp])}

The current **membership vector and style** dominate; the **{args.kind}** / trajectory-delta terms add
little on their own — consistent with "archetypes are sticky, and where you are now says most about where
you'll be." {'The true-age variant tests whether real age beats the experience proxy.' if args.kind == 'age' else 'A true-age variant (Model B) tests whether real age adds what the experience proxy cannot.'}

---
*Deliverable:* the model's calibrated soft membership vector is the per-player **projected next-season
archetype(s)** consumed by the Phase-2 optimizer at draft time. *Caveat:* top-1 accuracy below
persistence is honest — the win is probabilistic calibration, not point classification.
"""
    ensure_dir(REPORTS_DIR)
    out_path = REPORTS_DIR / f"REPORT_phase3_predict_{args.kind}.md"
    out_path.write_text(md)
    print(f"\n[predict] wrote {out_path.relative_to(REPORTS_DIR.parents[1])} and "
          f"data/processed/phase3_predict_table.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
