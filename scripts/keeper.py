"""Keeper-league assistant — predicted next-season rank + a prioritized keeper list.

Takes a CSV of the players you could keep and the pick each was drafted at, projects next season
with the per-position models, builds a value-over-replacement board for your league, and ranks
the keepers by surplus (= pick paid − projected board slot).

Usage:
    uv run python scripts/keeper.py --input my_keepers.csv --teams 12 --format sf

Input CSV columns: ``player,pick`` (optional ``position``). Covers RB/WR/QB; TE/K/DST and
unmatched names are listed as unscored.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from position_predictor.eval.keeper import build_board, evaluate_keepers  # noqa: E402
from position_predictor.eval.projection import project_positions  # noqa: E402
from position_predictor.utils.config import Config  # noqa: E402

DEFAULT_CONFIGS = ["config/football_rb.yaml", "config/football_wr.yaml", "config/football_qb.yaml"]


def main() -> int:
    p = argparse.ArgumentParser(description="Keeper-league predicted ranks + priority list.")
    p.add_argument("--input", required=True, help="CSV with columns: player, pick[, position].")
    p.add_argument("--teams", type=int, default=12, help="League size (8-16).")
    p.add_argument("--format", default="1qb", choices=["1qb", "sf", "2qb"],
                   help="QB format: 1qb / superflex / 2qb.")
    p.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS,
                   help="Per-position configs to project.")
    p.add_argument("--out", default="reports/keeper_board.csv")
    args = p.parse_args()

    if not (8 <= args.teams <= 16):
        p.error("--teams must be between 8 and 16")

    picks = pd.read_csv(args.input)
    picks.columns = [c.strip().lower() for c in picks.columns]
    if not {"player", "pick"}.issubset(picks.columns):
        p.error("input CSV must have at least 'player' and 'pick' columns")

    print(f"[keeper] projecting next season ({args.teams}-team {args.format}) …")
    proj = project_positions([Config.load(c) for c in args.configs])
    if proj.empty:
        print("[keeper] no projections produced — build features first (make features).")
        return 1

    board, replacement, starters = build_board(proj, teams=args.teams, fmt=args.format)
    season = int(proj["proj_season"].iloc[0])
    repl_str = ", ".join(f"{k} {v:.1f}" for k, v in replacement.items())
    print(f"[keeper] {season} replacement PPG by position: {repl_str}\n")

    ranked, unmatched = evaluate_keepers(board, picks)
    if ranked.empty:
        print("[keeper] none of the input players matched a projection.")
    else:
        show = ranked.rename(columns={"player_name": "player"})[
            ["player", "pos_rank", "proj_ppg", "proj_overall_rank", "pick", "surplus", "keep"]]
        show["surplus"] = show["surplus"].round(0).astype(int)
        print(f"Prioritized keepers (surplus = pick − projected board slot; {season} proj):\n")
        print(show.to_string(index=False))
        ranked.to_csv(args.out, index=False)
        print(f"\n[keeper] wrote {args.out}")

    if not unmatched.empty:
        print("\nUnscored (no model projection — TE/K/DST or name not matched):")
        print(unmatched[["player", "pick"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
