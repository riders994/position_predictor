"""Phase 3 — next-season archetype predictor: build the N->N+1 table, evaluate walk-forward, write report.

Baseline to beat is **persistence** (same archetype as last season). The honest question: does modeling
style trajectory + experience beat "sticky" on hard top-1 accuracy, and/or on the calibrated soft
membership distribution (log-loss) that Phase 2 consumes at draft time?

Compares two feature sets in one report: Model A (years-of-experience proxy, no fetch) and Model B (true
age from nba_api, included automatically once `make fetch-bio` has written the ages table).

Usage: uv run python scripts/predict_archetypes.py --config config/nba_archetypes.yaml
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
    args = p.parse_args()

    Config.load(args.config)
    table = build_predict_table(write=True)
    persist_full = round(float((table["arch"] == table["target_arch"]).mean()), 3)
    has_age = "age" in table.columns and table["age"].notna().mean() > 0.5
    print(f"[predict] {len(table)} player-season pairs · {table['athlete_id'].nunique()} players · "
          f"seasons N {int(table.season.min())}-{int(table.season.max())} · "
          f"persistence (full table) {persist_full:.3f} · age available: {has_age}")

    # Model A (experience proxy) always; Model B (true age) when the age table has been fetched.
    kinds = ["yoe"] + (["age"] if has_age else [])
    labels = {"yoe": "Model A (YOE)", "age": "Model B (true age)"}
    results = {k: walk_forward(table, kind=k) for k in kinds}
    imp = feature_importance(table, kind=kinds[-1])
    ra = results["yoe"]
    pm = ra["persistence"]

    print(f"\n[predict] walk-forward ({ra['eval_seasons'][0]}-{ra['eval_seasons'][-1]}, n={ra['n_eval']}):")
    print(f"   {'persistence':18s} acc={pm['accuracy']:.3f}  logloss={pm['log_loss']:.3f}  "
          f"brier={pm['brier']:.3f}")
    print(f"   {'marginal':18s} acc={ra['marginal']['accuracy']:.3f}  "
          f"logloss={ra['marginal']['log_loss']:.3f}")
    for k in kinds:
        m = results[k]["model"]
        print(f"   {labels[k]:18s} acc={m['accuracy']:.3f}  logloss={m['log_loss']:.3f}  "
              f"brier={m['brier']:.3f}  (acc_lift {results[k]['acc_lift_vs_persistence']:+.3f}, "
              f"ll_gain {results[k]['logloss_gain_vs_persistence']:+.3f})")
    print("\n[predict] top permutation-importance features:")
    for f in imp[:8]:
        print(f"   {f['feature']:16s} {f['importance']:+.4f}")

    # ---- report ----
    rows = [["persistence", f"{pm['accuracy']:.3f}", f"{pm['macro_f1']:.3f}", f"{pm['log_loss']:.3f}",
             f"{pm['brier']:.3f}"],
            ["marginal", f"{ra['marginal']['accuracy']:.3f}", f"{ra['marginal']['macro_f1']:.3f}",
             f"{ra['marginal']['log_loss']:.3f}", f"{ra['marginal']['brier']:.3f}"]]
    for k in kinds:
        m = results[k]["model"]
        rows.append([labels[k], f"{m['accuracy']:.3f}", f"{m['macro_f1']:.3f}", f"{m['log_loss']:.3f}",
                     f"{m['brier']:.3f}"])
    best = results[kinds[-1]]["model"]
    age_line = ""
    if "age" in kinds:
        ay, aa = results["yoe"]["model"], results["age"]["model"]
        age_line = (f"\n**Does true age beat the experience proxy?** Barely — Model B (age) lands at "
                    f"acc {aa['accuracy']:.3f} / log-loss {aa['log_loss']:.3f} vs Model A (YOE) "
                    f"{ay['accuracy']:.3f} / {ay['log_loss']:.3f}. Neither age nor experience cracks the "
                    f"top features. The plan's expectation that **age carries most of the lift is not "
                    f"supported**: archetype transitions are governed by *where a player is now* (current "
                    f"membership + style), not by age or one-year trajectory.\n")
    md = f"""# Phase 3 — Next-Season Archetype Predictor

Leak-safe **N→N+1** prediction of a player's archetype, returning players only. Features as-of season N:
current soft membership `p0..p11` (+ `top_prob`/`entropy`), 19 style z-features, one-year style
**trajectory** deltas, archetype tenure, and an experience/age term. Two feature sets are compared —
**Model A** uses a years-of-experience proxy (leak-safe, no fetch); **Model B** swaps in **true age**
(nba_api birthdates, {int(round(100 * table['age'].notna().mean()))}% covered). Target: archetype in N+1.
Evaluation is **walk-forward** (train only on transitions into earlier seasons); all rows are scored on
the same {ra['n_eval']} pooled pairs ({ra['eval_seasons'][0]}-{ra['eval_seasons'][-1]}).

## Walk-forward results

{_md_table(["predictor", "top-1 acc", "macro-F1", "log-loss", "Brier"], rows)}

> The models **do not beat persistence on hard top-1 accuracy** ({best['accuracy']:.3f} vs
> {pm['accuracy']:.3f}) — archetype membership is highly persistent, so "same as last season" is a very
> strong argmax baseline. But they **more than halve log-loss** ({best['log_loss']:.3f} vs
> {pm['log_loss']:.3f}): the models produce far better-calibrated soft membership vectors, which is
> exactly what Phase 2 consumes at draft time (projected category coverage needs a probability
> distribution, not a single hard label). **Marginal** (always the most common archetype) is the floor.
{age_line}
## What drives the prediction (permutation importance, {labels[kinds[-1]]})

{_md_table(["feature", "importance"], [[f["feature"], f"{f['importance']:+.4f}"] for f in imp])}

The current **membership vector and style** dominate; the experience/age and trajectory-delta terms add
little on their own — "archetypes are sticky, and where you are now says most about where you'll be."

---
*Deliverable:* the model's calibrated soft membership vector is the per-player **projected next-season
archetype(s)** consumed by the Phase-2 optimizer at draft time. *Caveats:* top-1 accuracy below
persistence is honest — the win is probabilistic calibration, not point classification. Every player-season
is **equal-weighted** (subject only to the 15-mpg/20-gp eligibility gate); a fantasy draft cares most
about top-of-draft players, so a value-weighted or top-N-by-value evaluation is a noted refinement (the
taxonomy and Phase-2 optimizer are already value-centric via the draft prior).
"""
    ensure_dir(REPORTS_DIR)
    out_path = REPORTS_DIR / "REPORT_phase3_predict.md"
    out_path.write_text(md)
    print(f"\n[predict] wrote {out_path.relative_to(REPORTS_DIR.parents[1])} and "
          f"data/processed/phase3_predict_table.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
