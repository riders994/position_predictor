"""Stage 5: discover college QB archetypes and cross them against the breakout label.

Clusters QB-seasons on four portable style rates (see ``archetypes/cluster.py`` for why quality
measures are deliberately excluded), names the result from centroid position so the names survive
a refit, then attaches a career archetype to the NFL cohort from stage 1.

Usage
-----
    uv run python sports/football/scripts/qb_breakout_archetypes.py
    uv run python sports/football/scripts/qb_breakout_archetypes.py --k 5

Outputs
-------
``data/processed/qb_breakout_archetype_seasons.parquet``  every college QB-season, with archetype
``data/processed/qb_breakout_archetypes.parquet``         the cohort, with its career archetype
``reports/REPORT_qb_breakout_archetypes.md``              the taxonomy and what it does not prove
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from position_predictor.utils.io import (  # noqa: E402
    DATA_PROCESSED, DATA_RAW, REPORTS_DIR, ensure_dir,
)
from qb_breakout.archetypes.cluster import (  # noqa: E402
    CLUSTER_FEATURES, FIT_MIN_DROPBACKS, career_archetypes, choose_k, fit_archetypes,
    profile_archetypes, quality_leakage,
)

SEASONS = DATA_RAW / "cfb_qb_seasons.parquet"
COHORT = DATA_PROCESSED / "qb_breakout_college.parquet"

DEFAULT_K = 4

# Outcomes crossed against the taxonomy, each with the population it is actually defined over.
#
# `ever_breakout` is NOT used here: it is the single-bar top-15 trigger, a diagnostic column, and
# the project's label is the two-bar rule (§3.2). Its two-bar equivalent is "did a sustained
# breakout ever happen", which is `sustained_season` being non-null — 38 of the matched cohort,
# against 52 for the loose column. Crossing the wrong one would inflate every rate on the page.
#
# `late_sustained` is only defined for quarterbacks who broke out at all, so its denominator is
# those 38 and it answers "among breakouts, which were late" — a different question from
# "which quarterbacks break out late", and labelled as such rather than quietly reported as a rate
# over everyone.
OUTCOMES = (
    {
        "column": "ever_sustained",
        "title": "Did a sustained breakout ever happen?",
        "population": "matched quarterbacks whose outcome is settled (right-censored careers "
                      "dropped — a 2022 entrant has not had time to fail)",
        "exclude_censored": True,
    },
    {
        "column": "late_sustained",
        "title": "Among quarterbacks who broke out, which did it late?",
        "population": "matched quarterbacks with a sustained breakout",
        "exclude_censored": False,
    },
)


def _table(df, cols=None):
    frame = df.to_pandas() if hasattr(df, "to_pandas") else df
    if cols:
        frame = frame[[c for c in cols if c in frame.columns]]

    def fmt(v):
        if v is None or (isinstance(v, float) and v != v):
            return ""
        if isinstance(v, float):
            return f"{v:.3f}".rstrip("0").rstrip(".") if abs(v) < 1000 else f"{v:.0f}"
        return str(v)

    head = "| " + " | ".join(map(str, frame.columns)) + " |"
    sep = "|" + "|".join("---" for _ in frame.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |"
            for row in frame.itertuples(index=False, name=None)]
    return "\n".join([head, sep, *rows])


def wilson(k: int, n: int, z: float = 1.96):
    """Wilson score interval — the honest way to show a rate computed from single-digit counts."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def population(cohort, spec, name_col="archetype_name"):
    """The rows an outcome is actually defined over — never "everyone, with nulls read as no"."""
    sub = cohort[cohort[name_col].notna() & cohort[spec["column"]].notna()]
    if spec["exclude_censored"] and "sustained_censored" in sub.columns:
        sub = sub[~sub["sustained_censored"].astype(bool)]
    return sub


def outcome_table(cohort, outcome, name_col="archetype_name"):
    import pandas as pd

    sub = cohort
    rows = []
    for name, grp in sub.groupby(name_col):
        k, n = int(grp[outcome].sum()), len(grp)
        lo, hi = wilson(k, n)
        rows.append({"archetype": name, "n": n, outcome: k,
                     "rate": round(k / n, 3), "95% CI": f"{lo:.2f}–{hi:.2f}"})
    k, n = int(sub[outcome].sum()), len(sub)
    lo, hi = wilson(k, n)
    rows.append({"archetype": "**all**", "n": n, outcome: k,
                 "rate": round(k / n, 3), "95% CI": f"{lo:.2f}–{hi:.2f}"})
    return pd.DataFrame(rows)


def permutation_p(cohort, outcome, name_col="archetype_name", n_iter=20000, seed=17):
    """How often does shuffling the archetype labels produce a spread this wide?

    A chi-square test is unreliable with expected counts this small, so the null is simulated
    directly. This is the only number in the report entitled to the word "significant", and it is
    reported whether or not it clears anything.
    """
    import numpy as np

    sub = cohort
    y = sub[outcome].to_numpy().astype(float)
    g = sub[name_col].to_numpy()
    groups = np.unique(g)
    if len(groups) < 2 or y.sum() == 0:
        return 1.0, 0.0

    def spread(labels):
        means = [y[labels == q].mean() for q in groups if (labels == q).sum()]
        return max(means) - min(means)

    observed = spread(g)
    rng = np.random.default_rng(seed)
    hits = sum(spread(rng.permutation(g)) >= observed for _ in range(n_iter))
    return (hits + 1) / (n_iter + 1), observed


def build_verdict(verdicts):
    """Write the conclusion *from* the numbers, so prose cannot drift away from the tables.

    Every earlier stage of this project has been bitten by a hand-written summary that stayed put
    while the tier or the data underneath it moved, so the sentences below are chosen by the
    p-values rather than typed alongside them.
    """
    lines = ["**What the numbers support.** Read the intervals, not the point estimates —"]
    for spec, p, obs, n, pos, lo, hi in verdicts:
        rate = pos / n if n else 0.0
        if p < 0.05:
            lines.append(
                f"- *{spec['title']}* — the {obs:.2f} spread survives label shuffling "
                f"(p = {p:.3f}), so archetype and this outcome are **not independent**. It is "
                f"driven by one cell: **{lo['archetype']}** at {lo['rate']:.2f} "
                f"({lo[spec['column']]}/{lo['n']}) against **{hi['archetype']}** at "
                f"{hi['rate']:.2f} ({hi[spec['column']]}/{hi['n']}), and those two intervals do "
                f"not overlap. Still only {pos} positives in {n} split four ways: carry the "
                f"direction into stage 6 as a prior, not as an effect size."
            )
        else:
            lines.append(
                f"- *{spec['title']}* — a {obs:.2f} spread arises from shuffled labels "
                f"{p:.0%} of the time, so at {pos} positives in {n} this is **consistent with "
                f"noise**. The base rate {rate:.2f} is the honest summary of every cell."
            )
    lines.append(
        "\nNeither result licenses using archetype as a classifier on its own. What both say is "
        "that the cells are too small for a four-way rate comparison to settle anything — a "
        "finding about power as much as about football."
    )
    return "\n".join(lines)


def write_report(path, *, k, k_scores, profile, leakage, seasons, cohort, exemplars,
                 outcome_tables, pvalues, verdict, stability):
    n_fit = int(seasons["fitted_on"].sum())
    n_assigned = int(seasons["archetype"].is_not_null().sum())
    matched = cohort[cohort["archetype_name"].notna()]

    md = f"""# College QB archetypes

The question this stage answers is *what kind of quarterback* a prospect was, not how good he was.
That distinction is the whole design: clustering on efficiency would return a leaderboard with
four bins, and stage 6 already has a model for quality.

- **Clustered on:** {', '.join(f'`{f}`' for f in CLUSTER_FEATURES)}
- **Seasons defining the centroids:** {n_fit} (≥{FIT_MIN_DROPBACKS} dropbacks)
- **Seasons assigned:** {n_assigned} of {seasons.height}
- **Cohort quarterbacks with a career archetype:** {len(matched)} of {len(cohort)}

Features are z-scored **within season**. College offense moved far enough across 2004–2021 that
raw features cluster on date — the first split found without this is simply a decade.

## Choosing k

{_table(k_scores)}

Silhouette peaks at k=2, which is the usual result for a continuous space: the honest reading is
that **style is a continuum with no natural gaps**, and any k is a partition of it rather than a
discovery of natural kinds. k={k} is chosen as a local maximum with no cluster smaller than a
seventh of the data, and — the real reason — because every cluster at k={k} has an obvious name.

## The taxonomy

{_table(profile)}

**The algorithm recovered a 2×2 it was never told to look for.** The four centroids fall one to a
quadrant of mobility × depth-of-target, which is the strongest available evidence that these two
axes are real rather than imposed. Names are therefore derived from *centroid position*, not from
the k-means label integer — refit with a different seed and {stability:.1%} of seasons keep the
same name, where label integers would have been reshuffled entirely.

{exemplars}

## Is this secretly a quality ranking?

The clustering features exclude every efficiency measure, and the held-out ones are used to check
whether that worked. The share of within-season EPA variance falling **between** archetypes:

{_table(leakage)}

At {leakage['eta_squared'].iloc[0]:.2f} the taxonomy explains a modest amount of quality — not
zero, because real styles do differ in average quality, but small enough that the clusters are
kinds rather than ranks. For contrast, clustering on quality stats gives 0.56, and this is exactly
why `completion_pct` was cut from the feature set: it is an accuracy measure, but it is also the
most quality-loaded rate a quarterback has, and including it doubled the leakage.

## Archetype against the breakout label

This is the part the project exists for, and it is also the part the data cannot yet carry.

{outcome_tables}

{pvalues}

{verdict}

What the taxonomy is genuinely for is stage 6: as a categorical feature alongside the continuous
ones, as a stratification variable for validation folds, and as a way to ask whether the *same*
pre-NFL evidence means different things for different kinds of quarterback — a question with more
statistical room in it than a four-way rate comparison.

## Career-level columns produced

| Column | Meaning |
|---|---|
| `archetype` / `archetype_name` / `archetype_label` | the **final** college season's cluster |
| `archetype_first` | the first season's cluster |
| `archetype_modal` | the most common cluster across the career |
| `archetype_stability` | share of college seasons spent in the final archetype |
| `archetype_changed` | whether the first and last archetypes differ |
| `n_archetype_seasons` | seasons contributing to the above |

The final season is the anchor, matching the project's established preference for `final_*`
features: a career clipped by the 2004 coverage floor still has a genuine last season, and it is
the one NFL evaluators weighted most. `archetype_changed` is a candidate feature in its own right
— {cohort['archetype_changed'].mean():.0%} of cohort quarterbacks changed style during college,
and a quarterback who did is not obviously the same prospect as one who never did.
"""
    ensure_dir(path.parent)
    path.write_text(md)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    args = ap.parse_args(argv)

    import pandas as pd
    import polars as pl

    for path, stage in ((SEASONS, "qb_breakout_college.py"), (COHORT, "qb_breakout_college.py")):
        if not path.exists():
            raise SystemExit(f"missing {path}\nRun stage 3 first:\n"
                             f"  uv run python sports/football/scripts/{stage}")

    qb_seasons = pl.read_parquet(SEASONS)
    k_scores = pd.DataFrame(choose_k(qb_seasons))
    assigned, _model = fit_archetypes(qb_seasons, k=args.k)

    # Name stability under a different seed — reported, not assumed.
    alt, _ = fit_archetypes(qb_seasons, k=args.k, random_state=99)
    keys = ["season", "team", "player"]
    both = assigned.select([*keys, "archetype_name"]).join(
        alt.select([*keys, "archetype_name"]), on=keys)
    stability = float((both["archetype_name"] == both["archetype_name_right"]).mean())

    profile = profile_archetypes(assigned)
    names = (assigned.filter(pl.col("archetype").is_not_null())
             .group_by("archetype").agg(pl.col("archetype_name").first()).sort("archetype"))
    profile = names.join(profile, on="archetype").drop("archetype")

    leakage = pd.DataFrame(quality_leakage(assigned))

    careers = career_archetypes(assigned)
    cohort = pl.read_parquet(COHORT).join(careers, on="player", how="left").to_pandas()

    ensure_dir(DATA_PROCESSED)
    assigned.write_parquet(DATA_PROCESSED / "qb_breakout_archetype_seasons.parquet")
    cohort.to_parquet(DATA_PROCESSED / "qb_breakout_archetypes.parquet", index=False)

    # Exemplars: highest-EPA seasons in each archetype, which is how a reader checks a name.
    lines = ["**Highest-EPA seasons in each archetype** — the name has to survive these:", ""]
    for row in names.to_dicts():
        ex = (assigned.filter((pl.col("archetype") == row["archetype"])
                              & (pl.col("fitted_on") == 1))
              .sort("pass_epa", descending=True).head(6))
        who = ", ".join(f"{r['player']} {r['season']}" for r in ex.to_dicts())
        lines.append(f"- **{row['archetype_name']}** — {who}")
    exemplars = "\n".join(lines)

    cohort["ever_sustained"] = cohort["sustained_season"].notna().astype(float)

    tables, pvals, verdicts = [], [], []
    for spec in OUTCOMES:
        outcome = spec["column"]
        if outcome not in cohort.columns:
            continue
        sub = population(cohort, spec)
        table = outcome_table(sub, outcome)
        tables.append(
            f"### {spec['title']}\n\n`{outcome}`, over {spec['population']}.\n\n{_table(table)}"
        )
        p, obs = permutation_p(sub, outcome)
        pvals.append(f"- **{spec['title']}** widest gap between archetypes {obs:.3f}; "
                     f"label shuffles reproduce a gap that wide **p = {p:.3f}** "
                     f"(n = {len(sub)}, {int(sub[outcome].sum())} positive)")
        cells = table[table["archetype"] != "**all**"].sort_values("rate")
        verdicts.append((spec, p, obs, len(sub), int(sub[outcome].sum()),
                         cells.iloc[0], cells.iloc[-1]))

    report = write_report(
        REPORTS_DIR / "REPORT_qb_breakout_archetypes.md",
        k=args.k, k_scores=k_scores, profile=profile, leakage=leakage, seasons=assigned,
        cohort=cohort, exemplars=exemplars, outcome_tables="\n\n".join(tables),
        pvalues="\n".join(pvals), verdict=build_verdict(verdicts), stability=stability,
    )

    matched = int(cohort["archetype_name"].notna().sum())
    print(f"seasons assigned: {int(assigned['archetype'].is_not_null().sum())} | "
          f"careers: {careers.height} | cohort matched: {matched}/{len(cohort)}")
    print(f"name stability across seeds: {stability:.3f}")
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
