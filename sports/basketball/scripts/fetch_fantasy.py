"""Phase 2 — fetch optimal-lineup success labels (max_pf) for the configured Fantrax leagues.

Usage:
    uv run python scripts/fetch_fantasy.py --config config/nba_archetypes.yaml
    uv run python scripts/fetch_fantasy.py --config config/... --weeks 1   # quick smoke test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nba_archetypes.data.fantrax import fetch_success_labels  # noqa: E402
from nba_archetypes.utils.config import Config  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Fetch max_pf optimal-lineup 9-cat success labels.")
    p.add_argument("--config", required=True)
    p.add_argument("--weeks", default=None, help="Week spec (default: full season). Use 1 to smoke-test.")
    p.add_argument("--methodology", default="hindsight", choices=["hindsight", "expected"])
    p.add_argument("--no-nash", action="store_true", help="Skip the Nash mutual-ceiling computation.")
    args = p.parse_args()

    cfg = Config.load(args.config)
    weeks = int(args.weeks) if (args.weeks and str(args.weeks).isdigit()) else args.weeks
    print(f"[fantasy] running max_pf ({args.methodology}, nash={not args.no_nash}) for "
          f"{len(cfg.get('fantasy.league_ids', []))} league(s) …")
    df = fetch_success_labels(cfg, weeks=weeks, methodology=args.methodology, nash=not args.no_nash)
    if df.empty:
        print("[fantasy] no rows (check league_ids / connectivity).")
        return 1
    print(f"[fantasy] {len(df)} team-seasons across {df.league_id.nunique()} league(s). "
          f"primary=M2 (both optimize): mean {df.primary_pf.mean():.1f}, "
          f"mean lineup_gap {df.lineup_gap.mean():.1f}")
    print(df.sort_values("primary_pf", ascending=False)
          [["league_id", "name", "actual_pf", "m2_pf", "m3_pf", "m1_pf", "lineup_gap"]]
          .head(10).to_string(index=False))
    print("[fantasy] wrote data/processed/fantrax_success.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
