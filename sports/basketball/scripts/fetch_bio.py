"""Phase-3 age source — fetch nba_api birthdates and derive per-season age (Model B input).

Maps our ESPN athlete_id -> nba_api id by name, fetches BIRTHDATE (cached; reruns only fetch missing
players), and writes per-(athlete_id, season) age to data/interim. Run once before `make predict --kind age`.

Usage: uv run python scripts/fetch_bio.py --config config/nba_archetypes.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nba_archetypes.data.bio import build_player_ages  # noqa: E402
from nba_archetypes.utils.config import Config  # noqa: E402
from nba_archetypes.utils.io import DATA_PROCESSED, read_parquet  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Fetch nba_api birthdates -> per-season age (Model B).")
    p.add_argument("--config", required=True)
    p.add_argument("--sleep", type=float, default=0.25, help="Seconds between nba_api calls.")
    p.add_argument("--refresh", action="store_true", help="Ignore the birthdate cache and refetch.")
    args = p.parse_args()

    Config.load(args.config)
    mem = read_parquet(DATA_PROCESSED / "nba_archetype_membership.parquet")
    ages, info = build_player_ages(mem, sleep=args.sleep, refresh=args.refresh)
    print(f"[bio] wrote data/interim/nba_player_ages.parquet · {len(ages)} player-seasons · "
          f"age coverage {info['coverage']:.1%} (nba_api matched {info['n_mapped']}, "
          f"{len(info['unmatched'])} unmatched, {len(info['ambiguous'])} ambiguous)")
    if info["unmatched"]:
        print(f"[bio] unmatched (no age unless ESPN-backfilled): {info['unmatched']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
