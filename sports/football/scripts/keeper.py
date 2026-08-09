"""Keeper-league assistant — predicted next-season rank + a prioritized keeper list.

Takes a CSV of the players you could keep and the pick each was drafted at, projects next season
with the per-position models, builds a value-over-replacement board for your league, and ranks
the keepers by surplus (= pick paid − projected board slot).

Point it at a league config so the board is built in your league's currency — the scoring format
decides the projections (a real retrain, not a rescale) and the roster shape decides replacement
level. Without ``--league`` it falls back to the ``--teams``/``--format`` shorthand and full PPR.

Usage:
    uv run python scripts/keeper.py --input my_keepers.csv --league config/leagues/my_2qb.yaml
    uv run python scripts/keeper.py --input my_keepers.csv --teams 12 --format sf

Input CSV columns: ``player,pick`` (optional ``position``). Covers QB/RB/WR/TE; K/DST and
unmatched names are listed as unscored.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from position_predictor.eval.keeper import build_board, evaluate_keepers  # noqa: E402
from position_predictor.eval.league import load_league  # noqa: E402
from position_predictor.eval.projection import project_positions  # noqa: E402
from position_predictor.utils.config import Config  # noqa: E402
from position_predictor.utils.io import REPORTS_DIR, ensure_dir  # noqa: E402

DEFAULT_CONFIGS = ["config/football_rb.yaml", "config/football_wr.yaml",
                   "config/football_qb.yaml", "config/football_te.yaml"]


def main() -> int:
    p = argparse.ArgumentParser(description="Keeper-league predicted ranks + priority list.")
    p.add_argument("--input", required=True, help="CSV with columns: player, pick[, position].")
    p.add_argument("--league", default=None,
                   help="League YAML (config/leagues/*.yaml) — sets scoring, teams, and slots.")
    p.add_argument("--teams", type=int, default=None,
                   help="League size (8-16). Ignored when --league is given.")
    p.add_argument("--format", default=None, choices=["1qb", "sf", "2qb"],
                   help="QB format shorthand: 1qb / superflex / 2qb. Ignored with --league.")
    p.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS,
                   help="Per-position configs to project.")
    p.add_argument("--out", default=None,
                   help="Output CSV (default reports/keeper_board[_<league>].csv).")
    args = p.parse_args()

    league = None
    if args.league:
        # The league config is the single source of truth; silently letting a stray --teams
        # override half of it is how you get a confident board for the wrong league.
        conflicting = [f for f, v in (("--teams", args.teams), ("--format", args.format))
                       if v is not None]
        if conflicting:
            p.error(f"{' and '.join(conflicting)} cannot be combined with --league "
                    f"(the league config already sets them)")
        try:
            league = load_league(args.league)
        except (ValueError, FileNotFoundError) as exc:
            p.error(str(exc))
        teams, roster, flex, scoring = (league.teams, league.starters, league.flex_positions,
                                        league.scoring)
        fmt = None
        label = league.label
    else:
        teams = args.teams if args.teams is not None else 12
        if not (8 <= teams <= 16):
            p.error("--teams must be between 8 and 16")
        fmt = args.format or "1qb"
        roster, flex, scoring = None, None, None
        label = f"{teams}-team {fmt} PPR"

    picks = pd.read_csv(args.input)
    picks.columns = [c.strip().lower() for c in picks.columns]
    if not {"player", "pick"}.issubset(picks.columns):
        p.error("input CSV must have at least 'player' and 'pick' columns")

    print(f"[keeper] projecting next season ({label}) …")
    proj = project_positions([Config.load(c) for c in args.configs], scoring=scoring)
    if proj.empty:
        print("[keeper] no projections produced — build features first (make features).")
        return 1

    kwargs = {"roster": roster, "flex_positions": flex} if league else {"fmt": fmt}
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    board, replacement, starters = build_board(proj, teams=teams, **kwargs)
    season = int(proj["proj_season"].iloc[0])
    repl_str = ", ".join(f"{k} {v:.1f}" for k, v in replacement.items())
    print(f"[keeper] {season} replacement PPG by position: {repl_str}")
    print(f"[keeper] started league-wide: "
          f"{', '.join(f'{k} {v}' for k, v in starters.items())}\n")

    ranked, unmatched = evaluate_keepers(board, picks)
    out = Path(args.out) if args.out else (
        ensure_dir(REPORTS_DIR)
        / (f"keeper_board_{league.name}.csv" if league else "keeper_board.csv"))
    if ranked.empty:
        print("[keeper] none of the input players matched a projection.")
    else:
        show = ranked.rename(columns={"player_name": "player"})[
            ["player", "pos_rank", "proj_ppg", "proj_overall_rank", "pick", "surplus", "keep"]]
        show["surplus"] = show["surplus"].round(0).astype(int)
        print(f"Prioritized keepers (surplus = pick − projected board slot; {season} proj):\n")
        print(show.to_string(index=False))
        ranked.to_csv(out, index=False)
        print(f"\n[keeper] wrote {out}")

    if not unmatched.empty:
        print("\nUnscored (no model projection — K/DST or name not matched):")
        print(unmatched[["player", "pick"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
