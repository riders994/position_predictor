"""Phase 2 — **roster optimizer**: build & validate coverage-maximizing draft rosters (incl. punt builds).

Uses the simulated field to calibrate the per-category win map (``Phi((cov-mu)/sigma)``), then:
  1. **validates** the optimizer in-sim — seats an optimizer team vs the manager/auto field for each
     punt build and reports its realized cat-win-rate lift + top-of-field rate;
  2. **demonstrates** the tool — prints the optimizer's balanced and best-punt rosters for the latest
     completed season (player names + projected per-category win profile).

Usage: uv run python scripts/optimize_phase2.py --config config/nba_archetypes.yaml
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nba_archetypes.eval.optimize import (  # noqa: E402
    CATS, evaluate_in_sim, field_coverage_stats, optimize_roster)
from nba_archetypes.eval.simulate import attach_prior, player_value_table  # noqa: E402
from nba_archetypes.utils.config import Config  # noqa: E402
from nba_archetypes.utils.io import DATA_INTERIM, DATA_PROCESSED, REPORTS_DIR, ensure_dir, read_parquet  # noqa: E402

# Punt builds to validate, chosen to span the 9-cat meta (elite punt_ft ... poor punt_tov).
BUILDS = ["balanced", "punt_ft", "punt_ft_ast", "punt_fg3m_ft", "punt_pts", "punt_fg", "punt_tov"]


def _md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _roster_names(pool_season, ids):
    lut = pool_season.set_index("athlete_id")["player_name"].to_dict()
    return [lut.get(i, str(i)) for i in ids]


def main() -> int:
    p = argparse.ArgumentParser(description="Phase-2 roster optimizer: validate + demonstrate.")
    p.add_argument("--config", required=True)
    p.add_argument("--n-leagues", type=int, default=25, help="Sim leagues per season for validation.")
    p.add_argument("--seed", type=int, default=1729)
    args = p.parse_args()

    cfg = Config.load(args.config)
    latest = int(cfg.get("data.latest_completed_season", 2026))
    seasons = list(range(2015, latest + 1))

    ps = read_parquet(DATA_INTERIM / "nba_player_seasons.parquet")
    values_prior = attach_prior(player_value_table(ps))
    table = read_parquet(DATA_PROCESSED / "phase2_simulated_composition.parquet")
    mu, sigma = field_coverage_stats(table, "cov_act")   # the field the optimizer is scored against

    # 1) validation
    summaries = []
    for build in BUILDS:
        _, _, s = evaluate_in_sim(values_prior, mu, sigma, seasons=seasons, n_leagues=args.n_leagues,
                                  punts=build, seed=args.seed)
        s["build"] = build
        summaries.append(s)
    summaries.sort(key=lambda s: s["lift"], reverse=True)

    print(f"[optimize] in-sim validation ({args.n_leagues} leagues/season, {min(seasons)}-{latest}):")
    for s in summaries:
        print(f"   {s['build']:14s} opt={s['opt_mean']:.4f}  field={s['field_mean']:.4f}  "
              f"lift={s['lift']:+.4f}  top-of-field={s['top_of_field_rate']:.3f}")

    # 2) demonstration on the latest season
    pool = values_prior[values_prior["season"] == latest]
    pool_season = ps[ps["season"] == latest]
    best_punt = next(s["build"] for s in summaries if s["build"] != "balanced")
    demos = {b: optimize_roster(pool, mu, sigma, n_rounds=13, punts=b) for b in ("balanced", best_punt)}
    for build, r in demos.items():
        print(f"\n[optimize] {latest} {build} roster (exp contested {r['exp_contested']:.3f}, "
              f"overall {r['exp_overall']:.3f}):")
        print("   " + ", ".join(_roster_names(pool_season, r["athlete_ids"][:8])) + ", …")

    # ---- report ----
    top_field = 1.0 / 12
    md = f"""# Phase 2 — Roster Optimizer (category-coverage builds)

The optimizer drafts the roster that maximizes **projected category coverage** for a chosen build. It
scores each candidate by the field-calibrated win map — modeling the field's per-category team coverage
as `N(mu_c, sigma_c)`, a roster's value is `mean_c Phi((cov_c - mu_c)/sigma_c)` over the *contested*
categories — using players' **prior-season** z-profiles (the draft-time projection). The Gaussian-CDF
shape makes it punt-aware: it stops piling onto locked categories and shores up winnable ones.

## In-sim validation

Each build seats one optimizer team against the manager/auto field ({args.n_leagues} leagues ×
{len(seasons)} seasons) and records its realized `cat_win_rate`. `top-of-field` is the share of leagues
the optimizer finishes best of 12 — **random ≈ {top_field:.3f}**.

{_md_table(["build", "optimizer", "field", "lift", "top-of-field"],
           [[s["build"], f"{s['opt_mean']:.4f}", f"{s['field_mean']:.4f}", f"{s['lift']:+.4f}",
             f"{s['top_of_field_rate']:.3f}"] for s in summaries])}

The optimizer beats the field, and **most on punt builds** — `punt_ft` is the strongest, consistent with
the 9-cat meta (concede one low-leverage percentage category to dominate the rest). Punting a
cheap-to-win category (`punt_tov`) backfires. The balanced lift is small by design: the shootout showed
draft-time projection is the binding constraint, so no draft-time optimizer can do much better than a
modest edge — the win comes from *build choice*, which is what this tool makes explicit.

## Example — {latest} optimizer rosters

{chr(10).join(f"**{b}** (expected contested {r['exp_contested']:.3f}, overall {r['exp_overall']:.3f}): "
              + ", ".join(_roster_names(pool_season, r['athlete_ids'])) for b, r in demos.items())}

Projected per-category win probability ({best_punt}):

{_md_table(["cat"] + [c.upper() for c in CATS],
           [["P(win)"] + [f"{demos[best_punt]['exp_win_by_cat'][c]:.2f}" for c in CATS]])}

---
*Caveats:* validation is in-sim (the simulator partly bakes in the archetype→category mapping), so
treat the lifts as *directional* and reality-check against real leagues. The optimizer uses prior-season
z as the projection; a proper next-season projection (Phase 3) is the upgrade that raises the ceiling.
"""
    ensure_dir(REPORTS_DIR)
    out_path = REPORTS_DIR / "REPORT_phase2_optimizer.md"
    out_path.write_text(md)
    print(f"\n[optimize] wrote {out_path.relative_to(REPORTS_DIR.parents[1])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
