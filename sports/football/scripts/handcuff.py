"""Handcuff selection — which backups to draft, ranked by starter injury risk.

For the upcoming season, projects every RB, identifies each team's starter + top backup, scores the
backup's *contingent upside* (the PPG it gains if the starter misses time × how likely that is), and
ranks the handcuffs worth drafting. Injury risk uses whichever signal wins a leak-safe backtest (the
§7.4 availability model vs transparent durability baselines). RB-only; model-only (no ECR/ADP).

Usage:
    uv run python scripts/handcuff.py                 # upcoming-season board
    uv run python scripts/handcuff.py --season 2025   # reconstruct a past draft year (leak-safe)
    uv run python scripts/handcuff.py --top 25 --signal durability_3yr
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from position_predictor.eval.handcuff import SIGNALS, render_markdown, run_handcuff  # noqa: E402
from position_predictor.utils.config import Config  # noqa: E402
from position_predictor.utils.io import REPORTS_DIR, ensure_dir  # noqa: E402

DEFAULT_CONFIG = "config/football_rb.yaml"


def main() -> int:
    p = argparse.ArgumentParser(description="Handcuff board — backups to draft by starter risk.")
    p.add_argument("--config", default=DEFAULT_CONFIG, help="RB position config.")
    p.add_argument("--season", type=int, default=None,
                   help="Draft season to board (default: upcoming season).")
    p.add_argument("--signal", choices=sorted(SIGNALS), default=None,
                   help="Override the backtest-chosen risk signal.")
    p.add_argument("--top", type=int, default=25, help="How many handcuffs to print.")
    p.add_argument("--out", default=None,
                   help="CSV path (default reports/handcuff_<season>.csv; .md alongside).")
    args = p.parse_args()

    config = Config.load(args.config)
    print("[handcuff] projecting RBs + backtesting risk signals …")
    res = run_handcuff(config, draft_season=args.season, signal=args.signal)
    if res.board is None or res.board.empty:
        print("[handcuff] no handcuff candidates — build features first (make features).")
        return 1

    if not res.backtest.empty:
        print(f"[handcuff] risk signal: {res.signal} (winner={res.winner}); backtest:")
        print(res.backtest.to_string(index=False))
    print(f"\nTop {args.top} handcuffs to draft ({res.season}):\n")
    cols = ["rank", "handcuff", "handcuff_pos_rank", "starter", "starter_pos_rank",
            "starter_exp_games_missed", "contingent_upside", "handcuff_value"]
    print(res.board.head(args.top)[cols].to_string(index=False))

    ensure_dir(REPORTS_DIR)
    csv_path = Path(args.out) if args.out else REPORTS_DIR / f"handcuff_{res.season}.csv"
    md_path = csv_path.with_suffix(".md")
    res.board.to_csv(csv_path, index=False)
    md_path.write_text(render_markdown(res, top=args.top))
    print(f"\n[handcuff] wrote {csv_path} + {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
