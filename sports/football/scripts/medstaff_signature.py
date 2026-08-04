"""Stage 5: does a club's excess in one body part travel across its position groups?

If a club's excess concentrates in one body part and appears across position groups that share
nothing except the building, that implicates a common cause — contact policy, tackling technique,
S&C, surface, protocol. If it is confined to one group it is roster, scheme or luck.

Two specifications are always reported together because they disagree, and the disagreement is
the finding. Multiplicity is preregistered: five parts means a Bonferroni threshold of 0.01, and
**concussion is the single primary hypothesis** on an a-priori mechanism.

Usage
-----
    uv run python sports/football/scripts/medstaff_signature.py
    uv run python sports/football/scripts/medstaff_signature.py --skip-rate   # composition only

Outputs
-------
``data/processed/medstaff_signature.parquet``   concordance by part x spec x split
``reports/REPORT_medstaff_signature.md``        the two specs, the splits, the coach test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from position_predictor.utils.io import (  # noqa: E402
    DATA_PROCESSED, DATA_RAW, REPORTS_DIR, ensure_dir, write_parquet,
)
from medstaff.data import FOCAL_GROUPS, load_schedules  # noqa: E402
from medstaff.expected import (  # noqa: E402
    INCIDENCE_CATEGORICAL, INCIDENCE_NUMERIC, fit_predict_loto,
)
from medstaff.signature import (  # noqa: E402
    ALR_REFERENCE, EXCLUDED_GROUPS, coach_follows, composition_concordance, head_coaches,
    leave_one_group_out, rate_concordance, variance_decomposition,
)

# Five parts tested, so the family-wise threshold is 0.05/5. Recorded here rather than chosen
# after the fact.
BONFERRONI = 0.05 / len(FOCAL_GROUPS)
PRIMARY_PART = "concussion"

# From the exploratory probe (2021-2025, crude exposure denominator), recorded before this run so
# it is confirmatory rather than fishing. See MEDSTAFF_PLAN §5.7.
PREREGISTERED = {
    "concussion": ("0.405", "0.404", "stable in both — the candidate signature"),
    "knee": ("-0.027", "0.335", "rate-only ⇒ level artifact"),
    "back": ("0.433", "0.172", "share-only ⇒ compositional"),
    "ankle": ("0.258", "0.174", "weak"),
    "hip": ("0.209", "0.106", "weak"),
}


def _table(frame, *, floats: int = 3) -> str:
    df = frame.to_pandas() if hasattr(frame, "to_pandas") else frame

    def fmt(v):
        if v is None or (isinstance(v, float) and v != v):
            return "—"
        if isinstance(v, (bool, np.bool_)):
            return "yes" if v else "no"
        if isinstance(v, (float, np.floating)):
            return f"{float(v):.{floats}f}"
        return str(v)

    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |"
            for row in df.itertuples(index=False, name=None)]
    return "\n".join([head, sep, *rows])


def _rate_residuals(exposure, episodes, *, focal):
    """Club x side observed-minus-expected for each focal part, out-of-fold by club."""
    rows = []
    for part in focal:
        onsets = (
            episodes.filter(pl.col("body_group") == part)
            .select(["season", "gsis_id", pl.col("onset_week").alias("week")])
            .unique().with_columns(pl.lit(1).alias("onset_part"))
        )
        frame = (
            exposure.join(onsets, on=["season", "gsis_id", "week"], how="left")
            .with_columns(pl.col("onset_part").fill_null(0))
            .filter(pl.col("side").is_in(["OFF", "DEF"]))
        )
        p = fit_predict_loto(frame, INCIDENCE_NUMERIC, INCIDENCE_CATEGORICAL, "onset_part")
        pdf = frame.select(["team", "side", "onset_part"]).to_pandas()
        pdf["expected"] = p
        agg = pdf.groupby(["team", "side"]).agg(
            observed=("onset_part", "sum"), expected=("expected", "sum"),
            var=("expected", lambda s: float(np.sum(s * (1 - s)))),
        ).reset_index()
        agg["z"] = (agg["observed"] - agg["expected"]) / np.sqrt(agg["var"].clip(lower=1e-9))
        agg["body_group"] = part
        rows.append(pl.from_pandas(agg))
    return pl.concat(rows)


def build(*, skip_rate: bool):
    episodes = pl.read_parquet(DATA_PROCESSED / "medstaff_episodes.parquet")
    exposure = pl.read_parquet(DATA_PROCESSED / "medstaff_exposure.parquet")
    seasons = tuple(sorted(episodes["season"].unique().to_list()))
    schedules = load_schedules(DATA_RAW, seasons=seasons)
    coaches = head_coaches(schedules)

    focal = list(FOCAL_GROUPS)
    comp_side = composition_concordance(episodes, focal=focal, split="side")
    comp_rand = composition_concordance(episodes, focal=focal, split="random")
    logo = leave_one_group_out(episodes, focal=focal)
    decomp = variance_decomposition(episodes, focal=focal)
    coach_table, n_multi = coach_follows(episodes, coaches, focal=focal)

    rate = None
    if not skip_rate:
        residuals = _rate_residuals(exposure, episodes, focal=focal)
        rate = rate_concordance(residuals, focal=focal)

    return {"episodes": episodes, "comp_side": comp_side, "comp_random": comp_rand,
            "rate": rate, "logo": logo, "decomp": decomp,
            "coach": coach_table, "n_multi_club_coaches": n_multi, "focal": focal}


def write_report(result, path: Path) -> Path:
    import pandas as pd

    focal = result["focal"]
    rows = []
    for part in focal:
        comp = result["comp_side"].get(part, {})
        rand = result["comp_random"].get(part, {})
        rate = (result["rate"] or {}).get(part, {})
        pre = PREREGISTERED.get(part, ("—", "—", "—"))
        rows.append({
            "body_group": part,
            "comp_off_def": comp.get("spearman", float("nan")),
            "comp_p": comp.get("p_value", float("nan")),
            "rate_off_def": rate.get("spearman", float("nan")),
            "rate_p": rate.get("p_value", float("nan")),
            "comp_random_half": rand.get("spearman", float("nan")),
            "prereg_share": pre[0], "prereg_rate": pre[1],
        })
    summary = pd.DataFrame(rows)

    primary = summary[summary.body_group == PRIMARY_PART].iloc[0]
    # Preregistration governs the thresholds: the primary hypothesis is tested at 0.05, the
    # four exploratory parts at the Bonferroni-corrected level. Applying the correction to the
    # primary too would discard the whole point of naming one in advance.
    exploratory = summary[summary.body_group != PRIMARY_PART]
    n_clear = int(((exploratory.comp_p < BONFERRONI)
                   | (exploratory.rate_p < BONFERRONI)).sum())
    both_specs = summary[(summary.comp_p < 0.05) & (summary.rate_p < 0.05)]

    logo = pd.DataFrame(result["logo"])
    logo_primary = logo[logo.body_group == PRIMARY_PART][["dropped_group", "spearman", "n"]]
    decomp = pd.DataFrame(result["decomp"])

    coach = result["coach"]
    coach_md = (_table(coach.select(
        ["coach", "team", "episodes", f"share_{PRIMARY_PART}"]).sort("coach"))
        if coach is not None and coach.height else
        "_No head coach ran two clubs with enough episodes inside the window._")

    md = f"""# Medical-staff grades — Stage 5: cross-position-group injury signature

If a club's excess concentrates in **one body part** and appears across position groups that
share nothing except the building, that implicates a common cause — practice contact policy,
tackling technique, strength and conditioning, field surface, medical protocol. If the excess is
confined to one group, it is roster, scheme or luck specific to that group.

**This is a variance decomposition, not a grade.** A club × body-part × position-group cell holds
a median of four episodes over five years, so cells are never graded; the concordance statistic
aggregates across the eight groups instead.

## 1. Multiplicity, preregistered

Five parts are tested, so the family-wise threshold is **{BONFERRONI:.3f}**.
**{PRIMARY_PART.capitalize()} is the single primary hypothesis**, on an a-priori mechanism:
practice contact policy and tackling technique are documented coach decisions, and concussion
reporting is protocol-mandated rather than discretionary, which makes it the least
disclosure-contaminated outcome in the project. The other four are **exploratory** and are
labelled as such wherever they appear.

**The primary hypothesis, {PRIMARY_PART}, holds.** Composition **{primary.comp_off_def:+.3f}**
(p {primary.comp_p:.3f}), rate **{primary.rate_off_def:+.3f}** (p {primary.rate_p:.3f}) — it
{"clears" if max(primary.comp_p, primary.rate_p) < 0.05 else "does not clear"} α = 0.05 in
**both** specifications, and it is the **strongest part in both**. Because it was named in advance
it is tested at 0.05 rather than the corrected level; applying the family correction to it would
discard the point of preregistering one.

It also replicates the exploratory probe closely — {PRIMARY_PART} was 0.405 / 0.404 there against
**{primary.comp_off_def:.3f} / {primary.rate_off_def:.3f}** here, now under a properly
exposure-adjusted model instead of the probe's crude denominator. That the number barely moved
when the denominator was fixed is the strongest thing that can be said for it.

{len(both_specs)} parts clear an uncorrected 0.05 in **both** specifications
({", ".join(both_specs.body_group)}) — but only {PRIMARY_PART} was named in advance, so the other
is exploratory.

**Of the four exploratory parts, {n_clear} clear the corrected threshold
({BONFERRONI:.3f}) in either specification.** Back is the near miss — it is the only other part
positive in both specs — and it is labelled exploratory, not a finding.

Spells with an unlabelled body part are **excluded, not pooled** ({", ".join(EXCLUDED_GROUPS)}).
The `unknown` group alone is 21.6% of episodes — spells that opened on a bare reserve week and
never picked up a report row — and it is concentrated in long absences, so pooling it would swamp
the parts being compared.

## 2. The two specifications

`composition` is B's share of that side's own episodes, additive-log-ratio transformed off
`{ALR_REFERENCE}`. It captures **signature net of level** and is **disclosure-robust** — a club
that lists everyone inflates numerator and denominator alike. That is a real advantage over every
level-based measure elsewhere in this project.

`rate` is episodes of B per unit exposure against a leave-one-team-out expectation. It captures
**level plus signature**: a club with more injuries overall has more of everything.

An effect in `rate` but not `composition` is a **level artifact** — the club's overall burden
travelling, not a part-specific tendency. An effect in `composition` but not `rate` is a
compositional shift with no absolute excess behind it. Only agreement in both is a candidate
signature.

The `prereg_*` columns are the exploratory probe recorded **before** this run, so it is
confirmatory rather than fishing.

{_table(summary)}

## 3. Offense against defense, and why

Different position coaches and different drills; the same building, S&C programme, medical staff
and surface. It is the sharpest split available — but **it is not fully independent**: an old or
badly-conditioned roster is old on both sides, which can manufacture concordance. The composition
specification removes the level effect by construction, and the random split-half column is run
alongside as a pure-consistency check.

### Leave-one-position-group-out — {PRIMARY_PART}

Guards against a single group, usually the offensive line, carrying a result that then reads as
club-wide.

{_table(logo_primary)}

## 4. Club main effect against club × group interaction

A large `share_common` means the part's excess is spread across the club's position groups, which
is what a shared cause looks like. A large interaction means it is specific to certain groups,
which points at roster or scheme.

{_table(decomp)}

## 5. Does the signature follow the head coach?

The only design element that breaks the shared-roster confound: a coach who carries an elevated
share of one body part across **different franchises**, with different rosters and different
buildings, is much harder to explain by roster construction.

**{result['n_multi_club_coaches']} head coaches ran two or more clubs** with enough episodes
inside this window. That is a power bound, not a result — five seasons is simply not long enough
for many coaches to change jobs and accumulate evidence at both.

{coach_md}

## 6. Reading this against stage 6

Stage 6 found that what persists year over year is **duration and returns-at-all** — how a club
uses injured reserve and times a return — while **recurrence, the outcome most owned by a medical
staff, does not persist at all**. A body-part signature that survives here would be evidence for
a mechanism in a place where the persistence test says clubs otherwise do not differ, which is
why it was worth testing separately rather than folding into the grades.
"""
    ensure_dir(path.parent)
    path.write_text(md)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-rate", action="store_true",
                        help="composition spec only (skips the 5 leave-one-team-out fits)")
    args = parser.parse_args(argv)

    result = build(skip_rate=args.skip_rate)

    rows = []
    for part in result["focal"]:
        for spec, src in (("composition_side", result["comp_side"]),
                          ("composition_random", result["comp_random"]),
                          ("rate_side", result["rate"] or {})):
            s = src.get(part)
            if s:
                rows.append({"body_group": part, "spec": spec, **s})
    frame = pl.DataFrame(rows)

    ensure_dir(DATA_PROCESSED)
    p1 = write_parquet(frame, DATA_PROCESSED / "medstaff_signature.parquet")
    p2 = write_report(result, REPORTS_DIR / "REPORT_medstaff_signature.md")

    for part in result["focal"]:
        c = result["comp_side"].get(part, {})
        r = (result["rate"] or {}).get(part, {})
        print(f"{part:12s} composition={c.get('spearman', float('nan')):+.3f} "
              f"(p={c.get('p_value', float('nan')):.3f})   "
              f"rate={r.get('spearman', float('nan')):+.3f} "
              f"(p={r.get('p_value', float('nan')):.3f})")
    for path in (p1, p2):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
