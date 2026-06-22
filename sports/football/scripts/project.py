"""Project the upcoming season for a position (the live board) -> reports/projections_<stem>.csv.

Run after the pipeline is built (features present). Usage:
    uv run python scripts/project.py --config config/football_rb.yaml [--model ridge] [--top 30]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from position_predictor.eval.projection import project_position  # noqa: E402
from position_predictor.utils.config import Config  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Project the upcoming season for one position.")
    p.add_argument("--config", required=True)
    p.add_argument("--model", default=None, help="Override projection model (default config/ridge).")
    p.add_argument("--top", type=int, default=30, help="How many to print.")
    args = p.parse_args()

    cfg = Config.load(args.config)
    out = project_position(cfg, model=args.model, write=True)
    if out.empty:
        print("[project] no projection produced (missing features?).")
        return 1
    season = int(out["proj_season"].iloc[0])
    print(f"[project] {cfg} — {season} projections (top {args.top}):")
    cols = ["proj_pos_rank", "player_name", "position", "proj_ppg"]
    print(out[cols].head(args.top).to_string(index=False))
    sport = cfg.get("experiment.sport", "sport")
    position = cfg.require("experiment.position")
    print(f"\n[project] wrote reports/projections_{sport}_{position}.csv".lower())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
