"""Calibrate the best-ball upside weight (lambda) against history.

Answers one question: given a player's projected points, does his week-to-week volatility carry
extra information about next season's realized best-ball value? Sweeps lambda per position over
walk-forward season pairs and reports the whole curve plus a paired significance test against
lambda = 0, so a flat curve is visible as flat rather than collapsed to an argmax.

Usage:
    uv run python scripts/bestball_calibrate.py
    uv run python scripts/bestball_calibrate.py --league config/leagues/underdog_bestball.yaml
    uv run python scripts/bestball_calibrate.py --from 2012 --to 2024
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from position_predictor.eval.bestball import (  # noqa: E402
    LAMBDA_GRID,
    fit_lambdas,
    weekly_fantasy_points,
)
from position_predictor.eval.league import MODELED_POS, load_league  # noqa: E402
from position_predictor.utils.io import DATA_RAW, REPORTS_DIR, ensure_dir  # noqa: E402

DEFAULT_LEAGUE = "config/leagues/underdog_bestball.yaml"
# Snap counts start in 2012 and the modern passing era begins around then; earlier seasons add
# noise more than signal for a volatility question.
DEFAULT_FROM, DEFAULT_TO = 2012, 2024


def main() -> int:
    p = argparse.ArgumentParser(description="Fit the best-ball upside weight from history.")
    p.add_argument("--league", default=DEFAULT_LEAGUE, help="League YAML defining the lineup.")
    p.add_argument("--from", dest="start", type=int, default=DEFAULT_FROM)
    p.add_argument("--to", dest="end", type=int, default=DEFAULT_TO)
    p.add_argument("--out", default=None,
                   help="JSON path (default reports/results/bestball_lambda_<league>.json).")
    args = p.parse_args()

    if args.start >= args.end:
        p.error("--from must be earlier than --to")

    league = load_league(args.league)
    print(f"[bestball] calibrating on {league.label} ({league.scoring}, {league.slot_summary()})")

    weekly = pd.read_parquet(DATA_RAW / "weekly.parquet")
    weekly_pts = weekly_fantasy_points(weekly, scoring=league.scoring)
    lambdas, detail = fit_lambdas(weekly_pts, league,
                                  seasons=range(args.start, args.end + 1))
    if detail.empty:
        print("[bestball] no season pairs available — is data/raw/weekly.parquet current?")
        return 1

    curve = detail.groupby(["position", "lambda"], as_index=False)["spearman"].mean()
    seasons = sorted(int(s) for s in detail["season"].unique())
    print(f"[bestball] {len(seasons)} walk-forward season pairs: "
          f"{seasons[0]}->{seasons[0] + 1} … {seasons[-1]}->{seasons[-1] + 1}\n")

    summary = {}
    print(f"{'pos':<5}{'lambda*':>9}{'rho(0)':>9}{'rho(l*)':>9}{'gain':>9}{'wins':>8}{'p':>8}")
    for pos in MODELED_POS:
        sub = detail[detail["position"] == pos]
        if sub.empty:
            continue
        lam = lambdas.get(pos, 0.0)
        base = sub[sub["lambda"] == 0.0].set_index("season")["spearman"]
        best = sub[sub["lambda"] == lam].set_index("season")["spearman"]
        diff = (best - base).dropna()
        pval = float(stats.ttest_1samp(diff, 0).pvalue) if len(diff) > 1 and diff.std() else 1.0
        print(f"{pos:<5}{lam:>9.2f}{base.mean():>9.4f}{best.mean():>9.4f}{diff.mean():>+9.4f}"
              f"{f'{int((diff > 0).sum())}/{len(diff)}':>8}{pval:>8.3f}")
        summary[pos] = {"lambda_argmax": lam, "spearman_at_zero": round(float(base.mean()), 4),
                        "spearman_at_argmax": round(float(best.mean()), 4),
                        "mean_gain": round(float(diff.mean()), 4),
                        "seasons_improved": int((diff > 0).sum()), "seasons": len(diff),
                        "p_value": round(pval, 4)}

    # A gain this small is indistinguishable from noise; say so rather than shipping the argmax.
    significant = {p: s for p, s in summary.items() if s["p_value"] < 0.05 and s["mean_gain"] > 0}
    print()
    if significant:
        print(f"[bestball] upside carries signal at: {', '.join(sorted(significant))} — "
              f"consider adopting those lambdas as defaults.")
    else:
        print("[bestball] NULL RESULT — no position's upside term beats lambda=0 at p<0.05. "
              "Rank best-ball boards by projected PPG; DEFAULT_LAMBDAS stays all-zero.")

    out = Path(args.out) if args.out else (
        ensure_dir(REPORTS_DIR / "results") / f"bestball_lambda_{league.name}.json")
    ensure_dir(out.parent)
    payload = {"league": league.name, "scoring": league.scoring, "lineup": league.slot_summary(),
               "seasons": seasons, "grid": list(LAMBDA_GRID), "per_position": summary,
               "adopted_lambdas": lambdas if significant else dict.fromkeys(MODELED_POS, 0.0),
               "significant": bool(significant)}
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
    curve.to_csv(out.with_name(f"bestball_lambda_curve_{league.name}.csv"), index=False)
    print(f"[bestball] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
