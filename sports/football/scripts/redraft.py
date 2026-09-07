"""Redraft assistant — check data, fetch, and project the coming season's draft board.

Runs the steps a redraft user wants for the upcoming season:
  (a) check nflverse has published the just-completed season,
  (b) refresh stale caches,
  (c) project the season once per (scoring format, market board) the configured leagues use,
  (d) turn those projections into one VORP-ranked draft board per league.

Leagues live in config/leagues/*.yaml (scoring, teams, started slots, flex eligibility). Two
leagues that share a scoring format *and* a market board share one pipeline pass, so adding a
league like an existing one is nearly free; a new format costs one dataset/features/fit pass per
position. A league that can start two QBs is benchmarked against the superflex board, so it gets
its own pass even when it shares a format.

Usage:
    uv run python scripts/redraft.py                  # every shipped league, draft season = now
    uv run python scripts/redraft.py --season 2026
    uv run python scripts/redraft.py --leagues config/leagues/my_2qb.yaml
    uv run python scripts/redraft.py --no-refresh     # use the cache as-is (skip fetch)
    uv run python scripts/redraft.py --top-qb 24 --top-rb 60 --top-wr 90 --top-te 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from position_predictor.data.benchmark import ecr_type_for_league  # noqa: E402
from position_predictor.eval.league import load_leagues  # noqa: E402
from position_predictor.eval.redraft import render_markdown, run_redraft  # noqa: E402
from position_predictor.utils.config import Config  # noqa: E402
from position_predictor.utils.io import REPORTS_DIR, ensure_dir  # noqa: E402

DEFAULT_CONFIGS = ["config/football_qb.yaml", "config/football_rb.yaml",
                   "config/football_wr.yaml", "config/football_te.yaml"]
DEFAULT_LEAGUES = ["config/leagues/ppr_1qb.yaml", "config/leagues/suz_1qb.yaml",
                   "config/leagues/my_2qb.yaml", "config/leagues/sar_1qb.yaml"
                   , "config/leagues/underdog_bestball.yaml"]
PRINT_COLS = ["proj_overall_rank", "player_name", "position", "proj_ppg", "vorp"]


def _parse_lambdas(values, parser):
    """``--bestball-lambda WR=0.3 TE=0.2`` → ``{"WR": 0.3, "TE": 0.2}``."""
    out = {}
    for item in values or []:
        if "=" not in item:
            parser.error(f"--bestball-lambda expects POS=VALUE; got {item!r}")
        pos, _, val = item.partition("=")
        try:
            out[pos.strip().upper()] = float(val)
        except ValueError:
            parser.error(f"--bestball-lambda value for {pos!r} is not a number: {val!r}")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Redraft draft-board projection for a new season.")
    p.add_argument("--season", type=int, default=None,
                   help="Draft season to project (default: current calendar year).")
    p.add_argument("--no-refresh", action="store_true",
                   help="Don't fetch; use the local cache as-is.")
    p.add_argument("--top-qb", type=int, default=None)
    p.add_argument("--top-rb", type=int, default=None)
    p.add_argument("--top-wr", type=int, default=None)
    p.add_argument("--top-te", type=int, default=None)
    p.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
    p.add_argument("--leagues", nargs="*", default=DEFAULT_LEAGUES,
                   help="League YAMLs to build boards for (default: all shipped leagues).")
    p.add_argument("--bestball-lambda", nargs="*", default=None, metavar="POS=VALUE",
                   help="Override the fitted upside weight, e.g. WR=0.3 (default: 0, see "
                        "eval/bestball.py — the fitted value is zero at every position).")
    p.add_argument("--top", type=int, default=60,
                   help="Rows shown in each report's overall board section (default 60).")
    p.add_argument("--out-dir", default=None,
                   help="Directory for the per-league CSV/MD (default: reports/).")
    args = p.parse_args()

    configs = [Config.load(c) for c in args.configs]
    try:
        leagues = load_leagues(args.leagues)
    except (ValueError, FileNotFoundError) as exc:
        p.error(str(exc))
    lambdas = _parse_lambdas(getattr(args, "bestball_lambda"), p)
    # Only positions the user actually named override the league-derived depth.
    top_n = {pos: n for pos, n in (("QB", args.top_qb), ("RB", args.top_rb),
                                   ("WR", args.top_wr), ("TE", args.top_te)) if n is not None}

    # A pass is one (scoring format, market board) pair — the board joins the key because a
    # two-QB league counts its rookies off the superflex chart. See eval.redraft.run_redraft.
    passes = sorted({(lg.scoring, ecr_type_for_league(lg)) for lg in leagues})
    print(f"[redraft] {len(leagues)} league(s): {', '.join(lg.name for lg in leagues)}")
    print(f"[redraft] {len(passes)} pipeline pass(es): "
          f"{', '.join(f'{s}/{e}' for s, e in passes)}")

    res = run_redraft(configs, draft_season=args.season, refresh=not args.no_refresh,
                      top_n=top_n or None, leagues=leagues, bestball_lambdas=lambdas or None)

    print(res.availability)
    if not res.ready:
        print(f"\n[redraft] STOP — season {res.feature_season} not fully published; "
              f"missing {res.availability.missing}. Re-run once nflverse releases it.")
        return 1
    if res.refreshed:
        print(f"[redraft] refreshed caches: {', '.join(res.refreshed)}")

    if not res.leagues or all(lb.board is None or lb.board.empty for lb in res.leagues):
        print("[redraft] no projections produced (missing features?).")
        return 1

    note = res.rookie_note or "rookie counts from market board"
    print(f"\n[redraft] {res.draft_season} draft boards "
          f"(features {res.feature_season}; rookie adjustment: {note})")

    out_dir = ensure_dir(Path(args.out_dir) if args.out_dir else REPORTS_DIR)
    for lb in res.leagues:
        lg = lb.league
        if lb.board is None or lb.board.empty:
            print(f"\n[redraft] {lg.name}: no board produced.")
            continue
        repl = ", ".join(f"{k} {v:.1f}" for k, v in lb.replacement.items())
        print(f"\n=== {lg.label} ({lg.scoring}, {lg.slot_summary()}) ===")
        print(f"replacement PPG: {repl}")
        print(lb.board.head(15)[PRINT_COLS].to_string(index=False))

        csv_path = out_dir / f"redraft_{res.draft_season}_{lg.name}.csv"
        md_path = out_dir / f"redraft_{res.draft_season}_{lg.name}.md"
        lb.board.to_csv(csv_path, index=False)
        md_path.write_text(render_markdown(res, lb, top=args.top))
        print(f"[redraft] wrote {csv_path.name} + {md_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
