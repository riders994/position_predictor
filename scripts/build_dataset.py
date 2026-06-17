"""Stage 2 — build the player-season dataset from the raw nflverse caches.

Joins weekly box scores + rosters into one ``(player_id, season)`` table for the configured
position, attaches the next-season PPG target and availability signals, materialises the
eligibility candidate grid, and writes it to ``data/interim/``.

Usage:
    uv run python scripts/build_dataset.py --config config/football_rb.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from position_predictor.data.build import build_dataset  # noqa: E402
from position_predictor.utils.config import Config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the player-season dataset (stage 2).")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config.")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    print(f"[build] {cfg}")

    df, result = build_dataset(cfg, write=True)
    span = f"{result.seasons[0]}–{result.seasons[-1]}" if result.seasons else "—"
    print(f"[build] rows={result.n_rows} cols={result.n_cols} seasons {span}")
    print(f"[build] rows with PPG target (active next season): {result.n_with_target}")
    print(f"[build] next-season status: {result.status_counts}")
    print(f"[build] wrote {result.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
