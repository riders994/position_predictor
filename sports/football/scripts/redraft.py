"""Redraft assistant — check data, fetch, and project the coming season's draft board.

Runs the three steps a redraft user wants for the upcoming season:
  (a) check nflverse has published the just-completed season,
  (b) refresh stale caches,
  (c) project the season -> top 20 QB / 50 RB / 75 WR / 24 TE (returning players within each top-N).

Usage:
    uv run python scripts/redraft.py                 # auto: draft season = current year
    uv run python scripts/redraft.py --season 2026
    uv run python scripts/redraft.py --no-refresh    # use the cache as-is (skip fetch)
    uv run python scripts/redraft.py --top-qb 24 --top-rb 60 --top-wr 90 --top-te 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from position_predictor.eval.redraft import DEFAULT_TOP_N, run_redraft  # noqa: E402
from position_predictor.utils.config import Config  # noqa: E402
from position_predictor.utils.io import REPORTS_DIR, ensure_dir  # noqa: E402

DEFAULT_CONFIGS = ["config/football_qb.yaml", "config/football_rb.yaml",
                   "config/football_wr.yaml", "config/football_te.yaml"]
PRINT_COLS = ["proj_pos_rank", "player_name", "position", "proj_ppg"]


def main() -> int:
    p = argparse.ArgumentParser(description="Redraft draft-board projection for a new season.")
    p.add_argument("--season", type=int, default=None,
                   help="Draft season to project (default: current calendar year).")
    p.add_argument("--no-refresh", action="store_true",
                   help="Don't fetch; use the local cache as-is.")
    p.add_argument("--top-qb", type=int, default=DEFAULT_TOP_N["QB"])
    p.add_argument("--top-rb", type=int, default=DEFAULT_TOP_N["RB"])
    p.add_argument("--top-wr", type=int, default=DEFAULT_TOP_N["WR"])
    p.add_argument("--top-te", type=int, default=DEFAULT_TOP_N["TE"])
    p.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
    p.add_argument("--out", default=None, help="Combined CSV path (default reports/redraft_<season>.csv).")
    args = p.parse_args()

    configs = [Config.load(c) for c in args.configs]
    top_n = {"QB": args.top_qb, "RB": args.top_rb, "WR": args.top_wr, "TE": args.top_te}

    res = run_redraft(configs, draft_season=args.season, refresh=not args.no_refresh, top_n=top_n)

    print(res.availability)
    if not res.ready:
        print(f"\n[redraft] STOP — season {res.feature_season} not fully published; "
              f"missing {res.availability.missing}. Re-run once nflverse releases it.")
        return 1
    if res.refreshed:
        print(f"[redraft] refreshed caches: {', '.join(res.refreshed)}")

    if res.board is None or res.board.empty:
        print("[redraft] no projections produced (missing features?).")
        return 1

    note = res.rookie_note or "rookie counts from market board"
    print(f"\n[redraft] {res.draft_season} draft board "
          f"(features {res.feature_season}; rookie adjustment: {note})\n")
    for s in res.summaries:
        sub = f" (top {s.requested_top_n} − {s.rookies_subtracted} rookies)" if s.rookies_subtracted \
            else f" (top {s.requested_top_n})"
        rows = res.board[res.board["position"] == s.position]
        print(f"=== {s.position}: {s.returned} returning players{sub} ===")
        print(rows[PRINT_COLS].to_string(index=False))
        print()

    out = Path(args.out) if args.out else ensure_dir(REPORTS_DIR) / f"redraft_{res.draft_season}.csv"
    res.board.to_csv(out, index=False)
    print(f"[redraft] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
