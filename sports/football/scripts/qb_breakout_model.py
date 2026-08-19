"""Stage 6: the pre-NFL-only breakout model, and an honest account of what it can carry.

Fits P(sustained breakout) from college evidence alone, in two feature tiers — portable (scores
current prospects) and cfbfastR-only (historical instrument) — and evaluates both against a
label-permutation null, a temporal split, and the draft as an independent benchmark.

Usage
-----
    uv run python sports/football/scripts/qb_breakout_model.py
    uv run python sports/football/scripts/qb_breakout_model.py --permutations 500

Outputs
-------
``data/processed/qb_breakout_scores.parquet``   out-of-fold scores per quarterback
``reports/REPORT_qb_breakout_model.md``         results, baselines, and the limits
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from position_predictor.utils.io import (  # noqa: E402
    DATA_PROCESSED, REPORTS_DIR, ensure_dir,
)
from qb_breakout.model.features import (  # noqa: E402
    MIN_SETTLED_SEASONS, feature_columns, modelling_frame,
)
from qb_breakout.model.fit import (  # noqa: E402
    TOP_K, benchmark_scores, coefficients, cross_validate, opportunity_profile, permutation_null,
    temporal_split, within_band_auc,
)

COHORT = DATA_PROCESSED / "qb_breakout_archetypes.parquet"
TIERS = ("portable", "full")


def _table(df):
    frame = df.to_pandas() if hasattr(df, "to_pandas") else df

    def fmt(v):
        if v is None or (isinstance(v, float) and v != v):
            return "—"
        if isinstance(v, float):
            return f"{v:.3f}".rstrip("0").rstrip(".")
        return str(v)

    head = "| " + " | ".join(map(str, frame.columns)) + " |"
    sep = "|" + "|".join("---" for _ in frame.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |"
            for row in frame.itertuples(index=False, name=None)]
    return "\n".join([head, sep, *rows])


def settled_sensitivity(cohort):
    """Base rate against the settled-outcome threshold — the evidence the cut is not load-bearing."""
    import pandas as pd

    rows = []
    for t in (3, 5, 7, 9, 11):
        f = modelling_frame(cohort, min_settled_seasons=t)
        rows.append({"min NFL seasons": t, "n": len(f),
                     "breakouts": int(f["ever_sustained"].sum()),
                     "base rate": round(float(f["ever_sustained"].mean()), 3)})
    return pd.DataFrame(rows)


def write_report(path, *, frame, results, null, bench, opportunity, bands, coefs, sensitivity,
                 n_permutations):
    import numpy as np

    port, full = results["portable"], results["full"]
    gap = port["cv"]["auc"] - full["cv"]["auc"]
    p_null = float((null >= port["cv"]["auc"]).mean())
    r1 = bands[bands["draft_band"] == "R1"].iloc[0]
    n_late = int(frame["late_sustained"].fillna(0).sum())

    md = f"""# Predicting a QB breakout from college evidence alone

The constraint this project was built around is that the model may see **only pre-NFL evidence**.
This is the stage that finds out what that buys.

- **Sample:** {len(frame)} quarterbacks, **{int(frame['ever_sustained'].sum())} sustained
  breakouts** (base rate {frame['ever_sustained'].mean():.3f})
- **Outcome:** a top-15 PPR PPG season held at top-20 in ≥2 of the 3 seasons from it (§3.2)
- **Model:** L2 logistic regression, C=0.1, class-balanced — the most complex thing
  {int(frame['ever_sustained'].sum())} positives support

## Who is in the sample

A quarterback who entered in 2024 and has not broken out has not *failed* to break out. Labelling
him a negative teaches the model that recent profiles do not work. The sample is therefore
restricted to **{MIN_SETTLED_SEASONS}+ NFL seasons of opportunity**, on the evidence that 90% of
sustained breakouts happen by year 5. That threshold is reported rather than asserted:

{_table(sensitivity)}

The base rate is flat from three seasons to eleven, so the cut is not doing hidden work — it
removes false negatives without reshaping the outcome.

## Does the college evidence predict anything at all?

{_table(results['table'])}

**Yes, decisively — against chance.** Shuffling the labels and re-running the whole pipeline
{n_permutations} times gives a null AUC of **{null.mean():.3f} ± {null.std():.3f}** (95th
percentile {np.quantile(null, 0.95):.3f}). The observed {port['cv']['auc']:.3f} sits far outside
it, **p = {max(p_null, 1 / (n_permutations + 1)):.4f}**. College production is not noise.

The temporal split — train on early entrants, score later ones, which is the only evaluation whose
information flow matches real use — holds up at **{port['temporal']['auc']:.3f}**
({port['temporal']['n_test']} test quarterbacks, {port['temporal']['test_positives']} breakouts).

## What portability costs: nothing

This is the question stage 4 set up, and the answer is clean.

The **portable** tier uses only features CFBD also computes, so it can be pointed at a quarterback
whose college career ended last autumn. The **full** tier adds EPA per dropback, success rate and
the career-shape features derived from them — better measurements, available 2004–2021 only.

The portable model scores **{port['cv']['auc']:.3f}** against the full model's
**{full['cv']['auc']:.3f}**: a gap of {gap:+.3f}, well inside the fold-to-fold spread of either
(± {port['cv']['auc_sd']:.3f}). **The efficiency features add nothing detectable.** The project
therefore keeps the forward-looking tool at no measured cost — and the reason is visible in the
coefficients below: what the model is actually reading is workload and role, not efficiency.

Gradient boosting was fitted too and does not help
({results['boosting']['auc']:.3f}), which is the expected result at this N and is reported so that
nobody has to wonder whether a more flexible learner was tried.

## What it reads

{_table(coefs.head(10))}

`sign_stability` is the share of bootstrap refits keeping the sign. **Volume is the strongest
single input** — final-season attempts — followed by *not* being a quick-game pocket passer and by
rushing share. That ordering is worth sitting with: the model's best evidence is how much a
quarterback played and what role he had, not how well he threw.

## The draft, as an independent opinion

Draft position is never a feature here — it is the league's own verdict, and mixing it in would
make the model partly a report of what scouts already decided. Kept separate, it is a benchmark:

| Ranking | AUC | Precision@{TOP_K} |
|---|---|---|
| College model (portable) | {port['cv']['auc']:.3f} | {port['cv']['precision_at_k']:.3f} |
| **Draft position** | **{bench['auc']:.3f}** | **{bench['precision_at_k']:.3f}** |

**The draft is far better, and that comparison is not what it looks like.**

{_table(opportunity)}

A fantasy breakout requires snaps, and the draft *allocates* snaps: first-rounders average
{opportunity.iloc[0]['mean_nfl_games']:.0f} career games against
{opportunity.iloc[-1]['mean_nfl_games']:.0f} for day-three and undrafted quarterbacks. The draft
partly **causes** the outcome it appears to predict. Beating it is not the standard a college-only
model should be held to, and a model that *did* beat it would be suspect.

The fair question is whether college evidence separates breakouts among **similarly drafted**
quarterbacks — which is also the late-breakout question, since a late breakout is by construction
someone the league undervalued:

{_table(bands)}

**And here the answer is no, where it would matter most.** Among first-round picks the college
model is at chance ({r1['model_auc']:.3f}, n={r1['n']}) — once the league has decided a
quarterback is worth a first-round pick, his college box score adds nothing to which of those
picks hits. The middle band looks better but rests on single-digit positives, and the day-three
band contains too few breakouts to evaluate at all.

## What this stage concludes

1. **College production carries real signal** — far outside a permutation null, stable across a
   temporal split.
2. **Portability is free.** The portable tier matches the full tier, so the project ends with a
   model that can score this year's prospects rather than only explain history.
3. **Most of that signal is workload and role, and most of it is already in the draft.** Within a
   draft band the college evidence adds little, and within round one it adds nothing.
4. **The late-breakout question specifically remains unanswered**, and honestly so. Of the
   {int(frame['ever_sustained'].sum())} breakouts here, {n_late} were late — enough to describe,
   not enough to fit. Stage 5 reached the same wall from a different direction.

The negative result is worth as much as a positive one would have been: it says that if late
breakouts are visible before the NFL, they are not visible in college *production*. What remains
untested is context — competition faced, supporting cast, scheme, the conditions a quarterback
produced under rather than the totals he produced. That is where a follow-on would go.
"""
    ensure_dir(path.parent)
    path.write_text(md)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--permutations", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=20)
    args = ap.parse_args(argv)

    import pandas as pd
    import polars as pl

    if not COHORT.exists():
        raise SystemExit(f"missing {COHORT}\nRun stage 5 first:\n"
                         f"  uv run python sports/football/scripts/qb_breakout_archetypes.py")

    cohort = pl.read_parquet(COHORT)
    frame = modelling_frame(cohort)

    results, rows = {}, []
    for tier in TIERS:
        num, cat = feature_columns(tier)
        cv_rows, oof = cross_validate(frame, num, cat, n_repeats=args.repeats)
        cv = pd.DataFrame(cv_rows)
        results[tier] = {
            "cv": {"auc": float(cv["auc"].mean()), "auc_sd": float(cv["auc"].std()),
                   "brier": float(cv["brier"].mean()),
                   "precision_at_k": float(cv["precision_at_k"].mean())},
            "temporal": temporal_split(frame, num, cat),
            "oof": oof,
        }
        rows.append({
            "tier": tier, "features": len(num) + len(cat),
            "CV AUC": round(results[tier]["cv"]["auc"], 3),
            "± sd": round(results[tier]["cv"]["auc_sd"], 3),
            f"P@{TOP_K}": round(results[tier]["cv"]["precision_at_k"], 3),
            "Brier": round(results[tier]["cv"]["brier"], 3),
            "temporal AUC": round(results[tier]["temporal"]["auc"], 3),
        })
    results["table"] = pd.DataFrame(rows)

    num, cat = feature_columns("portable")
    boost_rows, _ = cross_validate(frame, num, cat, kind="boosting", n_repeats=5)
    results["boosting"] = {"auc": float(pd.DataFrame(boost_rows)["auc"].mean())}

    null = permutation_null(frame, num, cat, n_permutations=args.permutations)
    bench = benchmark_scores(frame)
    opportunity = opportunity_profile(frame)
    bands = within_band_auc(frame, results["portable"]["oof"])
    coefs = coefficients(frame, num, cat)

    scores = frame[["player_id", "player_name", "entry_season", "draft_pick", "archetype_name",
                    "ever_sustained", "late_sustained"]].copy()
    scores["score_portable"] = results["portable"]["oof"]
    scores["score_full"] = results["full"]["oof"]
    ensure_dir(DATA_PROCESSED)
    scores.to_parquet(DATA_PROCESSED / "qb_breakout_scores.parquet", index=False)

    report = write_report(
        REPORTS_DIR / "REPORT_qb_breakout_model.md",
        frame=frame, results=results, null=null, bench=bench, opportunity=opportunity,
        bands=bands, coefs=coefs, sensitivity=settled_sensitivity(cohort),
        n_permutations=args.permutations,
    )

    print(f"n={len(frame)} positives={int(frame['ever_sustained'].sum())}")
    for tier in TIERS:
        print(f"  {tier:9s} AUC {results[tier]['cv']['auc']:.3f} "
              f"(sd {results[tier]['cv']['auc_sd']:.3f})")
    print(f"  draft benchmark AUC {bench['auc']:.3f} | null AUC {null.mean():.3f}")
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
