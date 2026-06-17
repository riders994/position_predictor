"""Stage 8 — walk-forward experiment (PROJECT_PLAN §6–§8).

Runs baselines + the era ensemble (× combiners) over the 10/20/30-yr windows and the fixed
test block, plus the NGS-block ablation and the availability model, and writes tidy result
tables to ``reports/results/``.

Usage:
    uv run python scripts/run_experiment.py --config config/football_rb.yaml
    uv run python scripts/run_experiment.py --config config/football_rb.yaml --fast
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from position_predictor.eval.experiment import run_experiment  # noqa: E402
from position_predictor.utils.config import Config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the walk-forward experiment (stage 8).")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config.")
    parser.add_argument("--fast", action="store_true",
                        help="Quick smoke config (fewer models/combiners/windows).")
    parser.add_argument("--models", nargs="*", help="Override candidate models.")
    parser.add_argument("--windows", nargs="*", type=int, help="Override history windows.")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    print(f"[experiment] {cfg}{' [fast]' if args.fast else ''}")

    res = run_experiment(cfg, write=True, fast=args.fast,
                         models=args.models, windows=args.windows)

    agg = res["ranking_aggregate"]
    g_star = int(cfg.get("eligibility.chosen_games_played", 4))
    if not agg.empty:
        at_star = agg[agg["cutoff_games"] == g_star].copy()
        print(f"[experiment] headline ranking @ g*={g_star} (Spearman mean ± sd across folds):")
        top = at_star.sort_values("spearman_mean", ascending=False)
        for _, r in top.head(12).iterrows():
            print(f"    {r['model_type']:12s} {r['model']:16s} w={int(r['window_years']):>2} "
                  f"{r['combine']:16s} ρ={r['spearman_mean']:.3f}±{r['spearman_sd']:.3f} "
                  f"P@12={r['precision_at_12_mean']:.2f} MAE={r['mae_mean']:.2f}")

    ab = res["ngs_ablation"]
    if not ab.empty:
        print("[experiment] NGS-block ablation (per window):")
        for _, r in ab.iterrows():
            print(f"    w={int(r['window_years']):>2} {r['variant']:12s} "
                  f"P@12={r.get('precision_at_12', float('nan')):.2f} "
                  f"ρ={r.get('spearman', float('nan')):.3f} keep={r.get('keep_ngs_block','')}")

    av = res["availability"]
    if not av.empty:
        m = av.groupby("model")[["games_mae", "clears_auc"]].mean()
        print("[experiment] availability model (mean across folds/windows):")
        for name, r in m.iterrows():
            print(f"    {name:22s} games_MAE={r['games_mae']:.2f} clears_AUC={r['clears_auc']:.3f}")

    print("[experiment] wrote reports/results/experiment_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
