"""Handcuff selection — which backups to draft, driven by starter injury risk.

Two modes, by position:
- **RB** (default): projects every RB, identifies each team's starter + top backup, and ranks the
  backups by *contingent upside* (PPG the backup gains if the starter misses time × how likely that
  is) — i.e. which handcuffs to draft.
- **QB / other**: a backup rarely inherits standalone value, so the output is simpler — a list of
  projected starters ranked by injury/availability risk, flagging who to draft a backup for.

Injury risk uses whichever signal wins a leak-safe backtest (the §7.4 availability model vs
durability baselines). Model-only (no ECR/ADP).

Scoring is adjustable: `--scoring half_ppr` (or `--league` to take it from a league config)
rebuilds the board against that format's target. It changes more than the magnitudes — the
"handcuff" is whoever projects as that team's RB2, so halving receptions can hand the role to a
different player. On the 2026 board it does exactly that for 2 of 27 starters (James Cook's
handcuff moves from pass-catching Ty Johnson to Ray Davis; Derrick Henry's from Justice Hill to
Keaton Mitchell), on top of rank shifts for 19 of 25 handcuffs and a lower mean contingent upside
(1.29 -> 1.14 PPG) since receptions are worth less.

Roster shape is deliberately ignored here: this is a within-team starter-vs-backup comparison, so
replacement level never enters. That is why the tool takes a scoring format rather than a league.

Usage:
    uv run python scripts/handcuff.py                                   # RB handcuff board
    uv run python scripts/handcuff.py --scoring half_ppr                # half-PPR board
    uv run python scripts/handcuff.py --league config/leagues/my_2qb.yaml
    uv run python scripts/handcuff.py --config config/football_qb.yaml  # QB injury-risk list
    uv run python scripts/handcuff.py --season 2025                     # past draft year (leak-safe)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from position_predictor.eval.handcuff import (  # noqa: E402
    SIGNALS,
    render_injury_markdown,
    render_markdown,
    run_handcuff,
    run_injury_risk,
)
from position_predictor.eval.league import load_league  # noqa: E402
from position_predictor.eval.projection import ensure_features  # noqa: E402
from position_predictor.scoring import DEFAULT_SCORING, RECEPTION_POINTS  # noqa: E402
from position_predictor.utils.config import Config  # noqa: E402
from position_predictor.utils.io import REPORTS_DIR, ensure_dir  # noqa: E402

DEFAULT_CONFIG = "config/football_rb.yaml"


def main() -> int:
    p = argparse.ArgumentParser(description="Handcuff tool — backups to draft by starter risk.")
    p.add_argument("--config", default=DEFAULT_CONFIG, help="Position config (RB board; else list).")
    p.add_argument("--season", type=int, default=None,
                   help="Draft season to board (default: upcoming season).")
    p.add_argument("--signal", choices=sorted(SIGNALS), default=None,
                   help="Override the backtest-chosen risk signal.")
    p.add_argument("--top", type=int, default=25, help="How many RB handcuffs to print (RB mode).")
    p.add_argument("--top-starters", type=int, default=32,
                   help="Projected starters to rank (QB/list mode).")
    p.add_argument("--scoring", choices=sorted(RECEPTION_POINTS), default=None,
                   help="Scoring format to board in (default: the config's, i.e. PPR).")
    p.add_argument("--league", default=None,
                   help="League YAML to take the scoring from (roster shape is not used here).")
    p.add_argument("--out", default=None,
                   help="CSV path (default reports/handcuff_<pos>_<season>[_<scoring>].csv; "
                        ".md alongside).")
    args = p.parse_args()

    if args.scoring and args.league:
        p.error("--scoring and --league both set the scoring format; pass only one")
    scoring = args.scoring
    if args.league:
        try:
            scoring = load_league(args.league).scoring
        except (ValueError, FileNotFoundError) as exc:
            p.error(str(exc))

    config = Config.load(args.config)
    if scoring:
        config = config.with_overrides({"target.scoring": scoring})
    position = config.require("experiment.position").upper()
    # A non-PPR format has its own dataset/features (artifact_stem namespaces them).
    ensure_features(config)

    if position == "RB":
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
        out_table, render = res.board, render_markdown(res, top=args.top)
    else:
        print(f"[handcuff] projecting {position}s + backtesting risk signals …")
        res = run_injury_risk(config, draft_season=args.season, signal=args.signal,
                              top_starters=args.top_starters)
        if res.risk_list is None or res.risk_list.empty:
            print("[handcuff] no projected starters — build features first (make features).")
            return 1
        if not res.backtest.empty:
            print(f"[handcuff] risk signal: {res.signal} (winner={res.winner}); backtest:")
            print(res.backtest.to_string(index=False))
        print(f"\n{position}s most likely to miss time — draft a backup ({res.season}):")
        print("(ranking is trustworthy; absolute games are biased low — read the tier)\n")
        print(res.risk_list.to_string(index=False))
        out_table, render = res.risk_list, render_injury_markdown(res, position=position)

    ensure_dir(REPORTS_DIR)
    # PPR keeps the historical filename; other formats get their own so boards don't clobber.
    suffix = "" if res.scoring == DEFAULT_SCORING else f"_{res.scoring}"
    csv_path = (Path(args.out) if args.out
                else REPORTS_DIR / f"handcuff_{position.lower()}_{res.season}{suffix}.csv")
    md_path = csv_path.with_suffix(".md")
    out_table.to_csv(csv_path, index=False)
    md_path.write_text(render)
    print(f"\n[handcuff] wrote {csv_path} + {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
