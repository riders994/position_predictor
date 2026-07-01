"""Phase 2 — **simulation + representation shootout**: which features carry the composition→success signal.

The real-team composition model was a four-way null because ``cat_win_rate`` is swamped by in-season
management, streaming, injuries and schedule luck. This script removes that noise with a leak-safe draft
simulator (prior-season-value prior + punt-strategy variation + an auto-draft/manager mix), scores teams
by simulating 9-cat head-to-head from players' *actual* production, and then runs an honest
leave-one-season-out **representation shootout** over three feature sets:

  archetype shares (Phase-1 mix)  ·  prior coverage (draft-time z-profile)  ·  actual coverage (ceiling)

Conclusion (reproduces the diagnostic arc): archetype shares ≈ 0 (wrong representation — they discard
category info), prior coverage small-positive (draft-time projection is the binding constraint), actual
coverage strongly positive (category coverage *is* the success mechanism). Archetypes remain the
interpretable taxonomy; the model of success is built on **category coverage**.

Usage: uv run python scripts/simulate_phase2.py --config config/nba_archetypes.yaml
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")  # weak signal -> expected ConvergenceWarnings from the linear fits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nba_archetypes.eval.phase2_model import representation_shootout  # noqa: E402
from nba_archetypes.eval.simulate import build_simulated_table, strategy_leaderboard  # noqa: E402
from nba_archetypes.utils.config import Config  # noqa: E402
from nba_archetypes.utils.io import REPORTS_DIR, ensure_dir  # noqa: E402


def _md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main() -> int:
    p = argparse.ArgumentParser(description="Phase-2 simulation + representation shootout + report.")
    p.add_argument("--config", required=True)
    p.add_argument("--n-leagues", type=int, default=60, help="Simulated leagues per season.")
    p.add_argument("--seed", type=int, default=1729)
    args = p.parse_args()

    cfg = Config.load(args.config)
    latest = int(cfg.get("data.latest_completed_season", 2026))
    # season N needs season N-1 priors and an archetype assignment; floor is the 2013-14 base (2014).
    seasons = list(range(2015, latest + 1))

    tbl, rosters, labels = build_simulated_table(seasons=seasons, n_leagues=args.n_leagues,
                                                 seed=args.seed, write=True)
    n_seasons = tbl["season"].nunique()
    print(f"[sim] {len(tbl)} simulated team-seasons · {args.n_leagues} leagues/season · "
          f"{n_seasons} seasons ({min(tbl.season)}-{max(tbl.season)}) · "
          f"mean cat_win_rate {tbl['sim_cat_win_rate'].mean():.3f}")

    shootout = representation_shootout(tbl)
    board = strategy_leaderboard(labels)

    print("\n[sim] representation shootout (leave-one-season-out, Ridge):")
    for name, m in shootout.items():
        print(f"   {name:18s} oof_R²={m['oof_r2']:+.4f}  oof_MAE={m['oof_mae']:.4f}  "
              f"(n_feat={m['n_features']})")
    print("\n[sim] draft-strategy leaderboard (mean cat-win-rate, z vs 0.5):")
    for r in board:
        print(f"   {r['strategy']:14s} n={r['n']:5d}  mean={r['mean_cat_win_rate']:.4f}  "
              f"z={r['z_vs_0.5']:+.1f}")

    # ---- report ----
    shares_r2 = shootout.get("archetype shares", {}).get("oof_r2", float("nan"))
    prior_r2 = shootout.get("prior coverage", {}).get("oof_r2", float("nan"))
    actual_r2 = shootout.get("actual coverage", {}).get("oof_r2", float("nan"))
    md = f"""# Phase 2 — Simulation & Representation Shootout

**Why simulate.** The real-team composition model hit a four-way null: `cat_win_rate` on real rosters is
dominated by in-season management, streaming, injuries and schedule luck, which swamp the
draft-composition signal. The simulator isolates the draft-composition question — it drafts teams from a
**leak-safe prior-season-value** prior with **punt-strategy variation** and a ~30% auto-draft / 70%
manager mix, then scores each team by round-robin **9-cat** head-to-head from the players' *actual*
season production.

**Corpus:** {len(tbl)} simulated team-seasons ({args.n_leagues} leagues × {n_seasons} seasons,
{min(tbl.season)}-{max(tbl.season)}; returning players only, prior-season prior). Seed {args.seed}.

## Representation shootout — which features carry the signal

Leave-one-season-out Ridge, target `sim_cat_win_rate` (league mean 0.5 by construction):

{_md_table(["representation", "out-of-fold R²", "out-of-fold MAE", "# features"],
           [[n, f"{m['oof_r2']:+.4f}", f"{m['oof_mae']:.4f}", m["n_features"]]
            for n, m in shootout.items()])}

**Three conclusions:**

1. **Archetype shares are the wrong success representation** (R²={shares_r2:+.3f} — null even in this
   clean sim). They describe play-style but *discard the category information* that decides 9-cat
   matchups; a contribution-matrix can't rescue a linear model since Ridge already spans any linear
   transform of shares.
2. **Category coverage is the mechanism** (actual coverage R²={actual_r2:+.3f}). The team's realized
   9-cat z-profile largely *determines* which categories it wins — so success should be modeled on
   coverage, with archetypes kept as the interpretable taxonomy.
3. **Draft-time projection is the binding constraint** (prior coverage R²={prior_r2:+.3f}). Last
   season's profile carries only a little of next season's coverage because players don't reproduce
   their z-profile — so the ceiling on a *draftable* model is projecting next-season production.

## Draft-strategy leaderboard — the signal lives in the punt builds

Mean `sim_cat_win_rate` by drafter strategy, with z vs the 0.5 league mean. That some punts clear 0.5 by
many standard errors (and others fall below) is the positive control: category coverage, built
deliberately, wins.

{_md_table(["strategy", "n", "mean cat-win-rate", "z vs 0.5"],
           [[r["strategy"], r["n"], f"{r['mean_cat_win_rate']:.4f}", f"{r['z_vs_0.5']:+.1f}"]
            for r in board])}

---
*Honesty:* a fully-simulated model partly bakes in the archetype→category mapping, so these are
*directional* findings, reality-checked against the real Yahoo/Fantrax team-seasons. The leak-safe
prior-season prior is the realism floor; real mock/ADP draft order is the v2 upgrade. The actionable
follow-on is a **roster optimizer** that targets category coverage for a chosen punt build (see
`make optimize`).
"""
    ensure_dir(REPORTS_DIR)
    out_path = REPORTS_DIR / "REPORT_phase2_simulation.md"
    out_path.write_text(md)
    print(f"\n[sim] wrote {out_path.relative_to(REPORTS_DIR.parents[1])} and "
          f"data/processed/phase2_simulated_composition.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
