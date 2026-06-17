"""Stage 1 — fetch nflverse data into the local Parquet cache.

Usage:
    uv run python scripts/fetch_data.py --config config/football_rb.yaml
    uv run python scripts/fetch_data.py --config config/football_rb.yaml --dry-run
    uv run python scripts/fetch_data.py --config config/football_rb.yaml \
        --datasets seasonal weekly snap_counts --overwrite
    uv run python scripts/fetch_data.py --config config/football_rb.yaml --include-large
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# allow running as a plain script (no install needed): add src/ to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from position_predictor.data.fetch import REGISTRY, fetch_all  # noqa: E402
from position_predictor.utils.config import Config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch nflverse data (stage 1).")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config.")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help=f"Subset of: {', '.join(REGISTRY)}")
    parser.add_argument("--include-large", action="store_true",
                        help="Also fetch heavy datasets (e.g. play-by-play).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Refetch even if a cache file exists.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show the fetch plan without contacting nflverse.")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    seasons = cfg.seasons()
    print(f"[fetch] {cfg}  seasons {seasons[0]}–{seasons[-1]} ({len(seasons)})")

    results = fetch_all(
        seasons,
        datasets=args.datasets,
        include_large=args.include_large,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )

    width = max(len(r.name) for r in results)
    n_err = 0
    for r in results:
        span = f"{r.seasons[0]}–{r.seasons[-1]}" if r.seasons else "—"
        shape = f"{r.n_rows}x{r.n_cols}" if r.n_rows is not None else ""
        print(f"  {r.name:<{width}}  {r.status:<8} {span:<11} {shape:<12} {r.message}")
        n_err += r.status == "error"

    print(f"[fetch] done: {len(results)} datasets, {n_err} error(s)")
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())
