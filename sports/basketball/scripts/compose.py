"""Phase 2 — roster archetype composition + join to success labels.

Pulls Fantrax rosters, name-matches players to Phase-1 archetypes, summarizes each fantasy team's
archetype composition, and joins the max_pf success labels (graded M2 > M3 > M1).

Usage: uv run python scripts/compose.py --config config/nba_archetypes.yaml
       uv run python scripts/compose.py --config config/... --no-fetch   # reuse cached rosters
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nba_archetypes.eval.compose import build_phase2_table, fetch_rosters  # noqa: E402
from nba_archetypes.utils.config import Config  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Phase-2 roster composition vs success labels.")
    p.add_argument("--config", required=True)
    p.add_argument("--no-fetch", action="store_true", help="Reuse cached rosters (skip Fantrax).")
    args = p.parse_args()

    cfg = Config.load(args.config)
    if not args.no_fetch:
        print("[compose] fetching Fantrax rosters …")
        r = fetch_rosters(cfg)
        print(f"[compose] rosters: {len(r)} rostered players across "
              f"{r.groupby(['league_id','team_id']).ngroups if len(r) else 0} teams, "
              f"seasons {sorted(r.season.unique()) if len(r) else '—'}")

    tbl, match_rate, unmatched = build_phase2_table(cfg)
    if tbl.empty:
        print("[compose] empty table — check rosters / success labels / membership.")
        return 1
    comp_cols = [c for c in tbl.columns if c.startswith("comp_")]
    print(f"[compose] {len(tbl)} teams · name-match rate {match_rate:.0%}")
    if len(unmatched):
        print(f"\n[compose] {len(unmatched)} UNMATCHED roster players (fix via fantasy.name_aliases "
              f"or check season coverage):")
        print(unmatched.sort_values("player_name")[["season", "player_name"]]
              .drop_duplicates().to_string(index=False))
    print("[compose] mean archetype composition (soft share), descending:")
    print(tbl[comp_cols].mean().sort_values(ascending=False).round(3).to_string())
    if "primary_pf" in tbl and tbl["primary_pf"].notna().any():
        corr = tbl[comp_cols].corrwith(tbl["primary_pf"]).sort_values(ascending=False)
        print("\n[compose] archetype-share correlation with success (M2 primary_pf):")
        print(corr.round(2).to_string())
    print("[compose] wrote data/processed/phase2_composition.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
