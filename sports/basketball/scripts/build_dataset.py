"""Stage 2 — pivot long player-season stats -> wide table (data/interim).

Usage: uv run python scripts/build_dataset.py --config config/nba_archetypes.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nba_archetypes.data.build import build_dataset  # noqa: E402
from nba_archetypes.utils.config import Config  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Assemble wide player-season table.")
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = Config.load(args.config)
    df = build_dataset(cfg)
    elig = int(df["eligible"].sum())
    print(f"[build] {cfg} rows={len(df)} cols={df.shape[1]} "
          f"seasons {int(df.season.min())}–{int(df.season.max())} eligible={elig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
