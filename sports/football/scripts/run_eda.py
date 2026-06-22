"""Stage 6 — exploratory data analysis (PROJECT_PLAN §3 ``[6] eda``).

Headless counterpart to the EDA notebooks: runs the coverage / distribution / target-stability
reads over the Stage-4 processed matrix and writes tidy tables + figures to ``reports/`` so the
analysis is reproducible without a Jupyter kernel. The notebooks import the same library code.

Usage:
    uv run python scripts/run_eda.py --config config/football_rb.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from position_predictor.eda.analyze import run_eda  # noqa: E402
from position_predictor.utils.config import Config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run EDA (stage 6).")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config.")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    print(f"[eda] {cfg}")

    tables = run_eda(cfg, write=True)

    uni = tables["universe"]
    print(f"[eda] player-seasons: {int(uni['n_player_seasons'].sum())} "
          f"over {int(uni['season'].min())}–{int(uni['season'].max())}")
    print("[eda] target coverage by season (last 6):")
    for _, r in uni.tail(6).iterrows():
        print(f"    {int(r['season'])}  n={int(r['n_player_seasons']):>3}  "
              f"target_cov={r['target_coverage']:.2f}  "
              f"(active={int(r['n_active_next'])} inj={int(r['n_injured_next'])} "
              f"ret={int(r['n_retired_next'])})")

    pooled = tables["target_stability"]
    p = pooled[pooled["season"] == -1].iloc[0]
    print(f"[eda] pooled year-over-year PPG: Spearman={p['spearman']:.3f} "
          f"Pearson={p['pearson']:.3f}  persistence MAE={p['persistence_mae']:.2f} "
          f"PPG  (n={int(p['n_pairs'])})")

    tier = tables["ngs_tier_coverage"]
    if not tier.empty:
        print("[eda] NGS rushing coverage by PPR tier (2016+):")
        for _, r in tier.iterrows():
            print(f"    {r['tier']:>6}  n={int(r['n']):>4}  "
                  f"has_ngs_rush={r['has_ngs_rush']:.2f}")

    rtm = tables["regression_to_mean"]
    if not rtm.empty:
        top = rtm.iloc[-1]
        print(f"[eda] mean reversion: top PPG quintile {top['mean_ppg_n']:.1f} -> "
              f"{top['mean_ppg_next']:.1f} next season (grand mean {top['grand_mean_next']:.1f}, "
              f"shrink {top['shrink']:.2f})")

    print("[eda] wrote reports/results/eda_* and reports/figures/eda_*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
