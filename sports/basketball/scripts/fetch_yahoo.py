"""Phase 2 augmentation — fetch the user's Yahoo 9-cat redraft history (rosters + labels).

Pulls each configured Yahoo league's season-long roster-weeks and category-win-rate success labels.
Heavy + rate-limited (~1800+ roster calls full); each league is cached so reruns resume.

Usage:
    uv run python scripts/fetch_yahoo.py --config config/nba_archetypes.yaml
    uv run python scripts/fetch_yahoo.py --config config/... --max-weeks 2   # quick smoke test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nba_archetypes.data.yahoo import fetch_yahoo  # noqa: E402
from nba_archetypes.utils.config import Config  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Fetch Yahoo season-long rosters + category-win labels.")
    p.add_argument("--config", required=True)
    p.add_argument("--max-weeks", type=int, default=None,
                   help="Cap regular-season weeks per league (smoke test; default: full season).")
    p.add_argument("--sleep", type=float, default=0.4, help="Seconds between Yahoo calls (throttle).")
    args = p.parse_args()

    cfg = Config.load(args.config)
    keys = cfg.get("fantasy_yahoo.league_keys", []) or []
    print(f"[yahoo] fetching {len(keys)} league(s) "
          f"({'full season' if not args.max_weeks else f'≤{args.max_weeks} wk'}) …")
    team_weeks, labels = fetch_yahoo(cfg, max_weeks=args.max_weeks, sleep=args.sleep)
    if labels.empty:
        print("[yahoo] no rows (check league_keys / token / connectivity).")
        return 1
    print(f"[yahoo] {len(labels)} team-seasons across {labels.league_key.nunique()} league(s); "
          f"{len(team_weeks)} player-weeks. mean cat-win-rate {labels.cat_win_rate.mean():.3f} "
          f"(should be ~0.5).")
    print("[yahoo] wrote data/processed/yahoo_team_weeks.parquet + yahoo_labels.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
