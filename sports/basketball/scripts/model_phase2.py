"""Phase 2 — model archetype composition -> 9-cat success and write the report.

Honest leave-one-league-out evaluation (does composition predict success out-of-sample?) +
the interpretable readout (which archetype tilts associate with winning, how stable, top-vs-bottom).

Usage: uv run python scripts/model_phase2.py --config config/nba_archetypes.yaml
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

# The signal is weak, so Lasso/coordinate-descent emits ConvergenceWarnings as it shrinks to ~0;
# that's expected here and just clutters the report run.
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nba_archetypes.eval.phase2_model import (  # noqa: E402
    compare_models, feature_cols, fit_coefficients, load_table, quartile_contrast)
from nba_archetypes.utils.config import Config  # noqa: E402
from nba_archetypes.utils.io import REPORTS_DIR, ensure_dir  # noqa: E402


def _md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main() -> int:
    p = argparse.ArgumentParser(description="Phase-2 composition -> 9-cat success model + report.")
    p.add_argument("--config", required=True)
    p.add_argument("--target", default="cat_win_rate")
    args = p.parse_args()

    Config.load(args.config)  # validates the config exists / is well-formed
    df = load_table()
    n_groups = df["league_key"].nunique()
    print(f"[model] {len(df)} team-seasons · {len(feature_cols(df))} archetype-share features · "
          f"{n_groups} league-seasons · target={args.target} "
          f"(mean {df[args.target].mean():.3f}, std {df[args.target].std():.3f})")

    models = compare_models(df, target=args.target)
    coefs = fit_coefficients(df, target=args.target)
    contrast = quartile_contrast(df, target=args.target)

    print("\n[model] leave-one-league-out generalization (vs mean baseline):")
    for name, m in models.items():
        print(f"   {name:18s} oof_R²={m['oof_r2']:+.3f}  oof_MAE={m['oof_mae']:.3f}  "
              f"within-league ρ={m['oof_spearman']:+.3f}")
    print("\n[model] Ridge standardized coefficients (archetype tilt -> success), top/bottom:")
    for r in coefs[:3] + coefs[-3:]:
        print(f"   {r['archetype']:26s} {r['coef_std']:+.4f}  (sign stable {r['sign_stability']:.0%})")

    # ---- report ----
    best_r2 = max(m["oof_r2"] for m in models.values())
    best_rho = max(m["oof_spearman"] for m in models.values() if m["oof_spearman"] == m["oof_spearman"])
    verdict = ("Composition shows **out-of-sample predictive signal**." if (best_r2 > 0.02 or best_rho > 0.15)
               else "Composition has **no reliable out-of-sample predictive power** at this sample "
                    "size — the value is directional (signs/contrast), not point prediction.")
    md = f"""# Phase 2 — Archetype Composition → 9-cat Success

**Corpus:** {len(df)} fantasy team-seasons across {n_groups} league-seasons (Yahoo 9-cat redraft,
2015-16…2023-24). **Features:** 12 weeks-weighted archetype soft shares (`comp_*`, sum≈1).
**Target:** `{args.target}` (mean {df[args.target].mean():.3f}, std {df[args.target].std():.3f}).

## Generalization (leave-one-league-season-out)

{_md_table(["model", "out-of-fold R²", "out-of-fold MAE", "within-league ρ"],
           [[n, f"{m['oof_r2']:+.3f}", f"{m['oof_mae']:.3f}", f"{m['oof_spearman']:+.3f}"]
            for n, m in models.items()])}

> {verdict} A mean-only baseline has R²=0 by construction; a model adds value only if it beats its MAE
> *or* shows a positive **within-league ρ** (correctly orders a held-out league's teams — the natural
> skill metric for a ranking/zero-sum target).

## Which archetype tilts associate with winning (Ridge, standardized)

Coefficients are *relative tilts away from the average roster mix* (compositional: shares sum to 1),
not independent effects. `sign stability` = fraction of league-resampled bootstraps keeping the sign —
**the trustworthy column at n≈{len(df)}**.

{_md_table(["archetype", "coef (std)", "sign stability"],
           [[r["archetype"], f"{r['coef_std']:+.4f}", f"{r['sign_stability']:.0%}"] for r in coefs])}

## Top vs bottom success quartile — mean composition

Descriptive and robust: how the most- and least-successful rosters are built, by share.

{_md_table(["archetype", "top quartile", "bottom quartile", "diff"],
           [[r["archetype"], r["top_q"], r["bottom_q"], f"{r['diff']:+.3f}"] for r in contrast])}

---
*Caveats:* n≈{len(df)} from one manager's league pool (selection); `cat_win_rate` is ~zero-sum within
a league; linear/Ridge won't capture **punt** builds (category concentration) — a concentration
feature + the archetype×category contribution matrix are the next refinements.
"""
    ensure_dir(REPORTS_DIR)
    out_path = REPORTS_DIR / f"REPORT_phase2_model_{args.target}.md"
    out_path.write_text(md)
    print(f"\n[model] wrote {out_path.relative_to(REPORTS_DIR.parents[1])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
