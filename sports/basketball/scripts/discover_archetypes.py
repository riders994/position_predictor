"""Phase 1 — discover player archetypes (soft GMM).

Usage:
    uv run python scripts/discover_archetypes.py --config config/nba_archetypes.yaml
    uv run python scripts/discover_archetypes.py --config config/... --select-k   # BIC/sil table
    uv run python scripts/discover_archetypes.py --config config/... --k 13
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nba_archetypes.archetypes.discover import discover_archetypes, select_k  # noqa: E402
from nba_archetypes.utils.config import Config  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Discover NBA archetypes via soft GMM.")
    p.add_argument("--config", required=True)
    p.add_argument("--select-k", action="store_true", help="Print the BIC/silhouette table only.")
    p.add_argument("--k", type=int, default=None, help="Override the number of archetypes.")
    args = p.parse_args()

    cfg = Config.load(args.config)
    if args.select_k:
        print("[archetypes] k-selection (BIC + silhouette):")
        print(select_k(cfg).to_string(index=False))
        return 0

    res = discover_archetypes(cfg, k=args.k)
    print(f"[archetypes] k={res.k}  BIC {res.bic:.0f}  assigned {len(res.membership)} player-seasons")
    sizes = res.profiles[["name", "size"]].sort_values("size", ascending=False)
    print(sizes.to_string(index=False))
    print("[archetypes] wrote data/processed/nba_archetype_membership.parquet + "
          "reports/REPORT_archetypes.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
