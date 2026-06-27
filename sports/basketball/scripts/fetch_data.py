"""Stage 1 — fetch NBA base data into data/raw (sportsdataverse / hoopR).

Usage:
    uv run python scripts/fetch_data.py --config config/nba_archetypes.yaml
    uv run python scripts/fetch_data.py --config config/... --include-large   # + shots/boxscore
    uv run python scripts/fetch_data.py --config config/... --datasets shots --overwrite
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nba_archetypes.data.fetch import REGISTRY, fetch_all  # noqa: E402
from nba_archetypes.utils.config import Config  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Fetch NBA base data (sportsdataverse) -> data/raw.")
    p.add_argument("--config", required=True)
    p.add_argument("--datasets", nargs="*", default=None,
                   help=f"Subset to fetch (default: non-large set). Known: {sorted(REGISTRY)}")
    p.add_argument("--include-large", action="store_true", help="Also fetch shots / boxscore.")
    p.add_argument("--overwrite", action="store_true", help="Refetch even if the cache exists.")
    args = p.parse_args()

    cfg = Config.load(args.config)
    seasons = cfg.seasons()
    print(f"[fetch] {cfg} seasons {seasons[0]}–{seasons[-1]}")
    results = fetch_all(seasons, datasets=args.datasets,
                        include_large=args.include_large, overwrite=args.overwrite)
    for r in results:
        size = f"{r.n_rows}x{r.n_cols}" if r.n_rows is not None else "-"
        print(f"  {r.name:22s} {r.status:8s} {size:>12s}  {r.message}")
    return 1 if any(r.status == "error" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
