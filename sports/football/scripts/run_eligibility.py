"""Stage 5 — derive the games / snap-share eligibility cutoff (PROJECT_PLAN §4.2).

Runs the split-half PPG reliability analysis + coverage/purity + snap-share reads over the
Stage-2 interim table and raw weekly data, picks the games cutoff, and writes the chosen rule,
the full candidate grid, and a figure to ``reports/``.

Usage:
    uv run python scripts/run_eligibility.py --config config/football_rb.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from position_predictor.eligibility.cutoff import derive_cutoff  # noqa: E402
from position_predictor.utils.config import Config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Derive eligibility cutoff (stage 5).")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config.")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    print(f"[eligibility] {cfg}")

    res = derive_cutoff(cfg, write=True)

    print("[eligibility] split-half PPG reliability:")
    for r in res["reliability_curve"]:
        rel = "  n/a" if r["reliability"] is None or r["reliability"] != r["reliability"] \
            else f"{r['reliability']:.3f}"
        print(f"    k={r['k']:>2}  reliability={rel}  (pool n={r['n_pool']})")

    reached = "reached" if res["reliability_target_reached"] else "NOT reached (max-reliability k)"
    print(f"[eligibility] chosen games cutoff g* = {res['chosen_games_played']} "
          f"(target {res['reliability_target']:g} {reached})")

    print("[eligibility] coverage vs purity by games cutoff:")
    for r in res["coverage_grid"]:
        print(f"    G>={r['games_cutoff']:>2}  n={r['n_eligible']:>4}  "
              f"coverage={r['coverage_points']:.3f}  excl_relevant={r['excluded_relevant']}")
    print("[eligibility] snap-share coverage (2012+):")
    for r in res["snap_grid"]:
        print(f"    snap>={r['snap_cutoff']:.2f}  n={r['n_eligible']:>4}  "
              f"coverage={r['coverage_points']:.3f}")
    print("[eligibility] wrote reports/results/eligibility_* and reports/figures/eligibility_*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
