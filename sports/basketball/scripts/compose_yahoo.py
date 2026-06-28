"""Phase 2 augmentation — Yahoo roster archetype composition + join to category-win-rate labels.

Weeks-weights each team's rostered players' archetype membership (season-long average) and joins the
Yahoo category-win-rate success labels -> the augmented Phase-2 modeling table.

Usage: uv run python scripts/compose_yahoo.py --config config/nba_archetypes.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nba_archetypes.data.yahoo import build_yahoo_phase2_table  # noqa: E402
from nba_archetypes.utils.config import Config  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Yahoo Phase-2 composition vs category-win-rate labels.")
    p.add_argument("--config", required=True)
    args = p.parse_args()

    cfg = Config.load(args.config)
    tbl, match_rate, unmatched = build_yahoo_phase2_table(cfg)
    if tbl.empty:
        print("[compose-yahoo] empty table — run fetch_yahoo first (need roster-weeks + labels).")
        return 1
    comp_cols = [c for c in tbl.columns if c.startswith("comp_")]
    print(f"[compose-yahoo] {len(tbl)} team-seasons · weeks-weighted name-match rate {match_rate:.0%}")
    if len(unmatched):
        top = (unmatched.groupby(["season", "player_name"])["weeks"].sum()
               .sort_values(ascending=False).head(25))
        print(f"\n[compose-yahoo] {unmatched['player_name'].nunique()} UNMATCHED players "
              f"(weight = player-weeks; fix via fantasy.name_aliases or check season coverage):")
        print(top.to_string())
    print("\n[compose-yahoo] mean archetype composition (weeks-weighted soft share), descending:")
    print(tbl[comp_cols].mean().sort_values(ascending=False).round(3).to_string())
    if "cat_win_rate" in tbl and tbl["cat_win_rate"].notna().any():
        corr = tbl[comp_cols].corrwith(tbl["cat_win_rate"]).sort_values(ascending=False)
        print("\n[compose-yahoo] archetype-share correlation with success (cat_win_rate):")
        print(corr.round(2).to_string())
    print("\n[compose-yahoo] wrote data/processed/phase2_yahoo_composition.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
