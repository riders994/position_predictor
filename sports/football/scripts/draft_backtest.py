"""Replay past drafts: the model's VORP board vs market drafting vs lookahead, on actual points.

    uv run python scripts/draft_backtest.py                       # full grid, default draws
    uv run python scripts/draft_backtest.py --scoring ppr --teams 12 --seasons 2023 --draws 2

Writes ``reports/results/draft_backtest_results.parquet`` (every draft, git-ignored) and the
committed ``reports/REPORT_draft_backtest.md``. See :mod:`position_predictor.eval.draft_backtest`.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _work(job):
    import warnings

    warnings.filterwarnings("ignore")
    from position_predictor.eval.draft_backtest import load_season_inputs, run_season

    spec, policies, slots = job
    board, proj, pts = load_season_inputs(spec.scoring, spec.season)
    t = time.time()
    rows, info = run_season(spec, board=board, proj=proj, weekly_pts=pts, slots=slots,
                            policies=policies)
    info["seconds"] = round(time.time() - t, 1)
    info["policies"] = ",".join(policies)
    return rows, info


def main(argv=None) -> int:
    import pandas as pd

    from position_predictor.data.adp import load_ffc_board
    from position_predictor.eval.draft_backtest import (BACKTEST_SEASONS, BACKTEST_TEAMS,
                                                        POLICY_NAMES, SeasonSpec, load_projections,
                                                        render_markdown, summarize)
    from position_predictor.utils.io import REPORTS_DIR, ensure_dir

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scoring", nargs="+", default=list(BACKTEST_SEASONS))
    ap.add_argument("--teams", nargs="+", type=int, default=list(BACKTEST_TEAMS))
    ap.add_argument("--seasons", nargs="+", type=int, default=None,
                    help="restrict to these seasons (default: every backtest season per format)")
    ap.add_argument("--draws", type=int, default=10, help="room draws per season")
    ap.add_argument("--lineup", nargs="+", default=["managed"], choices=["managed", "bestball"],
                    help="lineup rule(s) teams are scored under (default: managed)")
    ap.add_argument("--policies", nargs="+", default=[p for p in POLICY_NAMES
                                                      if not p.startswith("lookahead")])
    ap.add_argument("--lookahead", action="store_true",
                    help="also run lookahead_ranked (expensive) on --lookahead-slots")
    ap.add_argument("--lookahead-slots", nargs="+", type=int, default=[1, 5, 10])
    ap.add_argument("--lookahead-draws", type=int, default=4)
    ap.add_argument("--plan-draws", type=int, default=8)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--no-report", action="store_true",
                    help="write results only; leave the committed report untouched")
    args = ap.parse_args(argv)

    specs = [SeasonSpec(s, y, t, draws=args.draws, plan_draws=args.plan_draws, lineup=lineup)
             for lineup in args.lineup
             for s in args.scoring for y in BACKTEST_SEASONS[s]
             if args.seasons is None or y in args.seasons
             for t in args.teams]
    if not specs:
        print("nothing to run")
        return 1

    # Warm the caches serially so parallel workers never race to write them.
    for scoring in sorted({s.scoring for s in specs}):
        for season in sorted({s.season for s in specs if s.scoring == scoring}):
            load_ffc_board(season, scoring=scoring)
            load_projections(scoring, season)

    jobs = [(spec, list(args.policies), None) for spec in specs]
    if args.lookahead:
        jobs += [(replace(spec, draws=args.lookahead_draws), ["lookahead_ranked"],
                  [s for s in args.lookahead_slots if s <= spec.teams]) for spec in specs]

    rows, infos = [], []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futures = {ex.submit(_work, job): job for job in jobs}
        for fut in as_completed(futures):
            r, info = fut.result()
            rows.extend(r)
            infos.append(info)
            print(f"[{time.time() - t0:6.0f}s] {info['lineup']} {info['scoring']} {info['season']} "
                  f"{info['teams']}-team {info['policies']}: {len(r)} drafts "
                  f"{info.get('skipped', '')} ({info.get('seconds', 0)}s)", flush=True)

    if not rows:
        print("no drafts ran (every season skipped?)")
        return 1
    results = pd.DataFrame(rows)
    out = ensure_dir(REPORTS_DIR / "results") / "draft_backtest_results.parquet"
    results.to_parquet(out, index=False)
    print(f"wrote {out} ({len(results)} drafts)")

    summary = summarize(results)
    print(summary["gains"].round(2).to_string(index=False))
    if not args.no_report:
        note = (f"Lookahead ran on slots {args.lookahead_slots} with {args.lookahead_draws} draws "
                f"and {args.plan_draws} planning draws per candidate (it forks every candidate at "
                "each of its first 8 picks, ~100x a board draft)." if args.lookahead else "")
        report = REPORTS_DIR / "REPORT_draft_backtest.md"
        report.write_text(render_markdown(summary, infos, draws=args.draws, lookahead_note=note))
        print(f"wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
