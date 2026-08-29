"""Fetch the market benchmark — preseason ADP/ECR (PROJECT_PLAN §2.3, §7.4).

Downloads historical FantasyPros ECR, picks each season's latest preseason snapshot for the
configured position, maps it to ``gsis_id`` via the nflverse ID crosswalk (``data/raw/ids.parquet``
— run ``make fetch`` first), and writes a small committed reference table to ``data/external/``.

Defaults to the 1QB redraft-overall board. Pass ``--league`` to take the board that matches a
league's QB shape (a 2QB/superflex league pulls ``rsf``), or ``--ecr-type`` to name one directly;
non-default boards write to their own ``market_<sport>_<pos>_<type>.parquet`` cache.

Usage:
    uv run python scripts/fetch_benchmark.py --config config/football_rb.yaml
    uv run python scripts/fetch_benchmark.py --config config/football_qb.yaml \
        --league config/leagues/my_2qb.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from position_predictor.data.benchmark import (  # noqa: E402
    ECR_TYPES,
    benchmark_path,
    build_market_benchmark,
    ecr_type_for_league,
)
from position_predictor.eval.league import load_league  # noqa: E402
from position_predictor.utils.config import Config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch the ADP/ECR market benchmark.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config.")
    parser.add_argument("--league", help="League YAML whose QB shape picks the market board.")
    parser.add_argument("--ecr-type", choices=list(ECR_TYPES),
                        help="Market board to fetch (overrides --league).")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    ecr_type = args.ecr_type
    if ecr_type is None and args.league:
        ecr_type = ecr_type_for_league(load_league(args.league))
    print(f"[benchmark] {cfg}" + (f"  board={ecr_type}" if ecr_type else ""))

    out = build_market_benchmark(cfg, ecr_type=ecr_type, write=True)
    print(f"[benchmark] {len(out)} player-seasons across {out['season'].nunique()} seasons")
    for season, g in out.groupby("season"):
        d = g["scrape_date"].max()
        print(f"    {int(season)}  n={len(g):>3}  preseason scrape {str(d)[:10]}  "
              f"ECR top: {', '.join(str(int(r)) for r in g.nsmallest(3, 'market_ecr')['market_ecr'])}")
    path = benchmark_path(cfg.get("experiment.sport", "sport"),
                          cfg.require("experiment.position"), ecr_type)
    print(f"[benchmark] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
