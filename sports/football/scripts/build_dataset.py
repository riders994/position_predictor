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
from position_predictor.data.rookie_build import build_dataset as build_rookie_dataset  # noqa: E402
from position_predictor.data.team_build import build_dataset as build_team_dataset  # noqa: E402
from position_predictor.utils.config import Config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the player-season dataset (stage 2).")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config.")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    print(f"[build] {cfg}")

    # DST is a team-level entity (team_stats/schedules, not the player weekly frame); a rookie
    # config predicts a draft-year outcome from draft-day-known info, not a next-season roll
    # forward (data/rookie_build.py) — both are different Stage-2 builds, same output shape.
    is_dst = str(cfg.require("experiment.position")).upper() == "DST"
    is_rookie = cfg.get("experiment.cohort") == "rookie"
    build = build_team_dataset if is_dst else build_rookie_dataset if is_rookie else build_dataset
    df, result = build(cfg, write=True)
    span = f"{result.seasons[0]}–{result.seasons[-1]}" if result.seasons else "—"
    print(f"[build] rows={result.n_rows} cols={result.n_cols} seasons {span}")
    print(f"[build] rows with PPG target (active next season): {result.n_with_target}")
    print(f"[build] next-season status: {result.status_counts}")
    print(f"[build] wrote {result.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
