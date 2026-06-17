"""Fetch the market benchmark — preseason ADP/ECR (PROJECT_PLAN §2.3, §7.4).

Downloads historical FantasyPros ECR, picks each season's latest preseason redraft-overall
snapshot for the configured position, maps it to ``gsis_id`` via the nflverse ID crosswalk
(``data/raw/ids.parquet`` — run ``make fetch`` first), and writes a small committed reference
table to ``data/external/``.

Usage:
    uv run python scripts/fetch_benchmark.py --config config/football_rb.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from position_predictor.data.benchmark import build_market_benchmark  # noqa: E402
from position_predictor.utils.config import Config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch the ADP/ECR market benchmark.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config.")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    print(f"[benchmark] {cfg}")

    out = build_market_benchmark(cfg, write=True)
    print(f"[benchmark] {len(out)} player-seasons across {out['season'].nunique()} seasons")
    for season, g in out.groupby("season"):
        d = g["scrape_date"].max()
        print(f"    {int(season)}  n={len(g):>3}  preseason scrape {str(d)[:10]}  "
              f"ECR top: {', '.join(str(int(r)) for r in g.nsmallest(3, 'market_ecr')['market_ecr'])}")
    print("[benchmark] wrote data/external/market_*.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
