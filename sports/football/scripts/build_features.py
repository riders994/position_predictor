"""Stage 4 — engineer features into the processed modeling matrix.

Reads the Stage-2 interim player-season table (+ raw weekly/NGS for team-context and NGS
blocks), computes the block-organised feature set, and writes the processed parquet plus the
block→columns map used for era-aware feature selection (PROJECT_PLAN §6.3).

Usage:
    uv run python scripts/build_features.py --config config/football_rb.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from position_predictor.features.build import build_features  # noqa: E402
from position_predictor.utils.config import Config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build engineered features (stage 4).")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config.")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    print(f"[features] {cfg}")

    df, block_columns, path = build_features(cfg, write=True)
    n_feat = sum(len(v) for v in block_columns.values())
    print(f"[features] rows={df.shape[0]} cols={df.shape[1]} feature_cols={n_feat}")
    for block, cols in block_columns.items():
        print(f"  {block:<16} {len(cols):>2} cols")
    print(f"[features] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
