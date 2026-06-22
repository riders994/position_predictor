"""Postseason report — grade model / ECR / ADP against a completed season's actual finish.

Usage:
    uv run python scripts/postseason.py                 # auto: latest completed season
    uv run python scripts/postseason.py --season 2024
    uv run python scripts/postseason.py --no-refresh    # use the cache as-is
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from position_predictor.eval.postseason import build_postseason_report, render_markdown  # noqa: E402
from position_predictor.utils.config import Config  # noqa: E402
from position_predictor.utils.io import REPORTS_DIR, ensure_dir  # noqa: E402

DEFAULT_CONFIGS = ["config/football_qb.yaml", "config/football_rb.yaml", "config/football_wr.yaml"]


def main() -> int:
    p = argparse.ArgumentParser(description="Postseason report: model vs ECR vs ADP vs actuals.")
    p.add_argument("--season", type=int, default=None,
                   help="Completed season to grade (default: latest published).")
    p.add_argument("--no-refresh", action="store_true", help="Use the local cache as-is.")
    p.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
    p.add_argument("--out", default=None, help="Markdown path (default reports/postseason_<season>.md).")
    p.add_argument("--top", type=int, default=24, help="Rows per position in the board table.")
    args = p.parse_args()

    configs = [Config.load(c) for c in args.configs]
    res = build_postseason_report(configs, season=args.season, refresh=not args.no_refresh)

    print(res.availability)
    if not res.ready:
        print(f"\n[postseason] STOP — season {res.season} not fully published; "
              f"missing {res.availability.missing}. Try an earlier --season.")
        return 1

    md = render_markdown(res, top=args.top)
    out_md = Path(args.out) if args.out else ensure_dir(REPORTS_DIR) / f"postseason_{res.season}.md"
    out_md.write_text(md)
    out_csv = ensure_dir(REPORTS_DIR) / f"postseason_{res.season}.csv"
    res.board.to_csv(out_csv, index=False)

    # console summary
    import pandas as pd
    print(f"\n[postseason] {res.season} — rank accuracy vs actual finish:\n")
    summ = pd.DataFrame(res.summary)
    if not summ.empty:
        cols = [c for c in ["position", "source", "n", "spearman", "mean_abs_rank_err"]
                if c in summ.columns]
        print(summ[cols].to_string(index=False))
    for pos, m in res.adp_match.items():
        print(f"  ADP {pos}: matched {m.get('matched', 0)}/{m.get('ranked', 0)}"
              + (f" ({m['error']})" if m.get("error") else ""))
    print(f"\n[postseason] wrote {out_md} and {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
