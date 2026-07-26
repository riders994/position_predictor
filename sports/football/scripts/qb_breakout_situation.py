"""Stage 7: does the drafting situation — franchise, regime — matter beyond the pick?

Raw team breakout rates are unreportable at this sample size (a median of 4 quarterbacks and 1
breakout per franchise), so every result here is **observed minus expected given draft capital**,
against a simulated null, paired with how large an effect would have had to be to show up.

Usage
-----
    uv run python sports/football/scripts/qb_breakout_situation.py
    uv run python sports/football/scripts/qb_breakout_situation.py --gm-table path/to/gms.csv

Outputs
-------
``data/processed/qb_breakout_situation.parquet``  cohort with franchise + regime attached
``reports/REPORT_qb_breakout_situation.md``       the analysis and its power limits
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
from qb_breakout.model.features import modelling_frame  # noqa: E402
from qb_breakout.situation.regime import (  # noqa: E402
    MIN_QBS_PER_REGIME, attach_gm, attach_situation, detectable_effect, expected_from_draft,
    group_permutation_p, regime_table,
)

COHORT = DATA_PROCESSED / "qb_breakout_archetypes.parquet"

FACTORS = (
    ("draft_franchise", "Drafting franchise"),
    ("draft_coach", "Head coach at the draft"),
    ("draft_gm", "General manager at the draft"),
)


def _table(df):
    frame = df.to_pandas() if hasattr(df, "to_pandas") else df

    def fmt(v):
        if v is None or (isinstance(v, float) and v != v):
            return "—"
        return f"{v:.3f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)

    head = "| " + " | ".join(map(str, frame.columns)) + " |"
    sep = "|" + "|".join("---" for _ in frame.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |"
            for row in frame.itertuples(index=False, name=None)]
    return "\n".join([head, sep, *rows])


def factor_section(frame, col, title, expected):
    """One factor: the extremes, the whole-factor test, and what was detectable."""
    table = regime_table(frame, col, expected)
    if not len(table):
        return (f"### {title}\n\nNo {col} reached the {MIN_QBS_PER_REGIME}-quarterback floor.",
                None)

    import pandas as pd

    overall = group_permutation_p(frame, col, expected)
    power = detectable_effect(frame, col, expected)
    extremes = table if len(table) <= 10 else pd.concat([table.head(5), table.tail(5)])

    # With this many uncorrected comparisons, some will clear 0.05 with nothing behind them.
    # Saying how many are expected is the difference between a result and a coincidence.
    n_tests = len(table)
    hits = int((table["p_two_sided"] < 0.05).sum())
    expected_hits = round(0.05 * n_tests, 1)
    multiplicity = (
        "**Individually "
        + (f"{hits} regime{'s' if hits != 1 else ''} clear{'' if hits != 1 else 's'} p < 0.05, "
           f"against {expected_hits} expected by chance from {n_tests} uncorrected comparisons.**"
           if hits else
           f"no regime clears p < 0.05, against {expected_hits} expected by chance from "
           f"{n_tests} uncorrected comparisons.**")
    )

    verdict = (
        f"Across all {power['regimes_evaluated']} evaluated, the spread of observed-minus-expected "
        f"is **p = {overall['p_value']:.3f}** against random reassignment — "
        + ("**more variation than chance produces**, though see the power note below."
           if overall["p_value"] < 0.05 else
           "**exactly what chance produces.** There is no franchise or regime effect visible here "
           "once draft capital is accounted for.")
    )

    md = f"""### {title}

{_table(extremes)}

*(best and worst of {len(table)} regimes with ≥{MIN_QBS_PER_REGIME} quarterbacks; `diff` is
breakouts above what their draft picks predicted.)*

{multiplicity} {verdict}

**What would have been detectable.** A typical regime here drafted
{power['median_qbs_per_regime']} quarterbacks, expecting {power['expected_breakouts']} breakouts.
To clear a 5% threshold it would have needed **{power['breakouts_needed']}** — about
{power['extra_breakouts_needed']} extra breakouts above expectation from
{power['median_qbs_per_regime']} quarterbacks. Nothing subtler than that could have been found,
whether or not it is there."""
    return md, overall


def write_report(path, *, frame, sections, gm_note, n_drafted):
    md = f"""# Drafting situation: franchise and regime

Does *where* a quarterback landed matter beyond *how highly* he was picked? The question has been
open since the project's first prompt — "player archetypes, drafting situations, and other
factors" — and this stage answers it as far as the arithmetic allows, which is not very far.

## Why there are no team breakout rates in this report

{n_drafted} drafted quarterbacks with a settled outcome, spread across 32 franchises: a **median
of 4 quarterbacks and 1 breakout each**. Seven franchises have zero, one has three. A table of raw
team rates would show a 0%–60% spread, every point of it noise, and it would be the most
screenshot-friendly thing in the repository. So it is not computed.

What replaces it is **observed minus expected**, where expected comes from draft capital alone —
each quarterback's out-of-fold P(breakout) given his pick, summed per regime. Teams differ hugely
in the picks they spend on quarterbacks, so a raw comparison is mostly a comparison of draft
position. This asks the question actually worth asking: *given the capital they spent, did any
regime get more out of quarterbacks than the pick predicted?*

The null is simulated by drawing each quarterback independently from his own expected probability
— no normal approximation, which would be wrong at these counts.

## Results

{sections}

## General managers

{gm_note}

## What this stage concludes

The honest summary is that **this question cannot be answered with 177 quarterbacks**, and the
detectable-effect numbers say so quantitatively rather than as a hedge. A franchise would have had
to produce roughly double its expected breakouts across its entire draft history to register, and
no plausible real coaching or front-office effect is that large.

That is not the same as saying situation does not matter. It says that *this* outcome — a career
event with 35 instances — is the wrong measuring instrument for it. A situation effect would be
far better tested against something that happens often: snaps earned in years one to three,
starts, or the fantasy points a quarterback actually scored, all of which have a data point per
season rather than one per career.

**The pick already contains much of what a team decision is.** §4.4 showed draft position predicts
this outcome at AUC 0.890 largely because it allocates opportunity. A franchise's influence on a
quarterback's career mostly *is* the pick it spent — and that is measured, not missing.
"""
    ensure_dir(path.parent)
    path.write_text(md)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gm-table", type=str, default=None,
                    help="CSV with columns season,team,gm (nflverse publishes no GM data)")
    args = ap.parse_args(argv)

    import pandas as pd
    import polars as pl

    if not COHORT.exists():
        raise SystemExit(f"missing {COHORT}\nRun stage 5 first:\n"
                         f"  uv run python sports/football/scripts/qb_breakout_archetypes.py")

    frame = modelling_frame(pl.read_parquet(COHORT))
    frame = attach_situation(frame)

    if args.gm_table:
        gm = pd.read_csv(args.gm_table).rename(columns={"gm": "draft_gm"})
        frame = attach_gm(frame, gm)
        gm_note = (f"Attached from `{args.gm_table}` — "
                   f"{int(frame['draft_gm'].notna().sum())} of {len(frame)} quarterbacks matched.")
    else:
        frame["draft_gm"] = None
        gm_note = (
            "**Not analysed, because there is no data for it.** nflverse publishes head coaches "
            "(via schedules, 1999+) but not general managers, and no free structured "
            "GM-by-team-season table exists. Head coach is used above as the available regime "
            "proxy — an imperfect one, since a coach and a general manager can disagree about a "
            "quarterback and often do.\n\n"
            "`situation/regime.py::attach_gm` takes a hand-supplied `(season, team, gm)` table and "
            "the script accepts `--gm-table`, so this is a one-CSV job whenever a list is "
            "available. Given the power arithmetic above it would not change the conclusion: "
            "general-manager tenures are *shorter* than franchise histories, so the cells would be "
            "smaller still.")

    expected = expected_from_draft(frame)
    frame["expected_from_draft"] = expected

    sections, results = [], {}
    for col, title in FACTORS:
        if col == "draft_gm" and frame["draft_gm"].isna().all():
            continue
        md, overall = factor_section(frame, col, title, expected)
        sections.append(md)
        results[col] = overall

    ensure_dir(DATA_PROCESSED)
    frame.to_parquet(DATA_PROCESSED / "qb_breakout_situation.parquet", index=False)

    n_drafted = int(frame["draft_franchise"].notna().sum())
    report = write_report(
        REPORTS_DIR / "REPORT_qb_breakout_situation.md",
        frame=frame, sections="\n\n".join(sections), gm_note=gm_note, n_drafted=n_drafted,
    )

    print(f"n={len(frame)} drafted={n_drafted} "
          f"coach matched={int(frame['draft_coach'].notna().sum())}")
    for col, res in results.items():
        if res:
            print(f"  {col:16s} dispersion p={res['p_value']:.3f}")
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
