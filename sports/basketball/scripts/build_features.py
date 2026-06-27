"""Stage 3 — engineer play-style features (data/processed).

Usage: uv run python scripts/build_features.py --config config/nba_archetypes.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nba_archetypes.features.build import BLOCKS, build_features  # noqa: E402
from nba_archetypes.utils.config import Config  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Engineer style features + within-season z-scores.")
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = Config.load(args.config)
    df = build_features(cfg)
    n_feat = sum(len(v) for v in BLOCKS.values())
    arch_pool = int((df["eligible"] & df["era_use_for_archetypes"]).sum())
    print(f"[features] {cfg} rows={len(df)} style_features={n_feat}")
    for blk, cols in BLOCKS.items():
        print(f"  {blk:12s} {len(cols)} cols")
    print(f"[features] Phase-1 clustering pool (eligible & E2+E3): {arch_pool} player-seasons")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
