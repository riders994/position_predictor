"""Phase-2 draft intel — per-archetype value→wins coefficients ("archetype value guide") + report.

Descriptive companion to the coverage optimizer: which archetypes convert roster value into 9-cat wins,
and which are value traps. (Round-by-round archetype *selection weights* were tested and lose in-sim —
see the report; this ships only the intel.)

Usage: uv run python scripts/archetype_value.py --config config/nba_archetypes.yaml
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nba_archetypes.eval.archetype_value import (  # noqa: E402
    build_exposure_table, classify, exposure_coefficients)
from nba_archetypes.utils.config import Config  # noqa: E402
from nba_archetypes.utils.io import REPORTS_DIR, ensure_dir  # noqa: E402


def _md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main() -> int:
    p = argparse.ArgumentParser(description="Phase-2 archetype value guide (draft intel) + report.")
    p.add_argument("--config", required=True)
    p.add_argument("--n-leagues", type=int, default=60)
    p.add_argument("--seed", type=int, default=1729)
    args = p.parse_args()

    cfg = Config.load(args.config)
    latest = int(cfg.get("data.latest_completed_season", 2026))
    table = build_exposure_table(seasons=list(range(2015, latest + 1)), n_leagues=args.n_leagues,
                                 seed=args.seed, write=True)
    prior = exposure_coefficients(table, kind="prior")
    actual = exposure_coefficients(table, kind="actual")
    guide = classify(prior, actual)

    print(f"[archetype-value] {len(table)} simulated team-seasons · value-weighted exposure → cat-win-rate")
    print(f"\n{'archetype':24s} {'draft coef':>10s} {'sign%':>6s} {'actual':>8s} {'top-bot Δ':>9s}  label")
    for r in guide:
        ac = f"{r['actual_coef']:+.3f}" if r["actual_coef"] is not None else "  -  "
        print(f"{r['archetype']:24s} {r['coef_std']:+10.4f} {r['sign_stability']:6.0%} {ac:>8s} "
              f"{r['top_minus_bottom']:+9.2f}  {r['label']}")

    prio = [r["archetype"] for r in guide if r["label"] == "PRIORITIZE"]
    traps = [r["archetype"] for r in guide if r["label"] in ("trap", "avoid")]
    md = f"""# Phase 2 — Archetype Value Guide (draft intel)

**Which archetypes convert roster *value* into 9-cat wins — and which are traps.** Built by regressing
simulated `cat_win_rate` on each team's **value-weighted archetype exposure** (`Σ max(value,0)·membership`
over the roster — magnitude kept, *not* normalized shares). Corpus: {len(table)} simulated team-seasons
({args.n_leagues} leagues × {table['season'].nunique()} seasons). Standardized Ridge coefficient =
value→wins efficiency; **draft coef** uses draft-time (prior-season) value and is the actionable column,
**actual** is the post-hoc ceiling; `sign%` = season-bootstrap sign stability; `top-bot Δ` = extra raw
exposure of top- vs bottom-quartile teams.

## The guide

{_md_table(["archetype", "draft coef", "sign%", "actual coef", "top−bot Δexp", "label"],
           [[r["archetype"], f"{r['coef_std']:+.4f}", f"{r['sign_stability']:.0%}",
             (f"{r['actual_coef']:+.3f}" if r["actual_coef"] is not None else "—"),
             f"{r['top_minus_bottom']:+.2f}", r["label"]] for r in guide])}

- **PRIORITIZE — {', '.join(prio) or 'none'}:** stably positive draft-time value→wins; winning rosters
  concentrate the most value here.
- **TRAP / avoid — {', '.join(traps) or 'none'}:** either a stably negative draft-time weight, or (the
  classic *trap*) an archetype that pays off on *realized* value but whose draft-time weight is a wash —
  it looks valuable (raw scoring) yet its 9-cat category profile doesn't convert (e.g., volume scorers
  who bleed TOV / FG%).

## Why this is *intel*, not a draft engine

Round-by-round archetype **selection weights** were built and evaluated in-sim: a picker scoring players
by `projected_value × (weight · archetype)` with diminishing returns **lost to the field** (lift −0.02),
and even break-even static weighting only tied it. Balancing across archetypes sacrifices the
value/coverage **concentration** (punt builds) that actually wins 9-cat. So the draft **engine** stays
the category-coverage optimizer (`make optimize`); these coefficients are a companion read for *which
archetypes to spend on*, not a mechanical picker. The draft-time coefficients are also small in absolute
terms — the Phase-3 projection ceiling caps how much any draft-time signal can deliver.

---
*Method note:* value-weighted **exposure** (magnitude) reaches oof R²≈0.30 on actual value vs ≈0 for
equal-weighted shares — confirming archetype composition carries success signal only when weighted by
player value *and* left un-normalized. See REPORT_phase2_simulation.md for the representation shootout.
"""
    ensure_dir(REPORTS_DIR)
    out_path = REPORTS_DIR / "REPORT_archetype_value.md"
    out_path.write_text(md)
    print(f"\n[archetype-value] wrote {out_path.relative_to(REPORTS_DIR.parents[1])} and "
          f"data/processed/phase2_archetype_value_exposure.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
