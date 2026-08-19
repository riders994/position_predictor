"""Link college QB production to the NFL cohort.

Same hazard as the recruiting join (``link.py``) and the same conservative posture, but a
different shape of evidence. Here we have two weak keys instead of one:

- **Name**, which cfbfastR carries but without any player ID.
- **School**, which nflverse spells differently from ESPN ("Michigan St." vs "Michigan State",
  "Ole Miss" vs "Mississippi") and which undrafted QBs do not have at all.

School is therefore used as a **tiebreaker, not a filter**. Requiring it to match would throw
away real players over spelling, and spelling variance is not random — it is worse for smaller
programs, which is exactly where late-round and undrafted QBs come from. The filter is
name-plus-plausible-timing; school only decides between candidates that survive it.
"""

from __future__ import annotations

import re

from .link import normalize_name

# A QB's last college season falls 1-5 years before he enters the NFL: normally the year before,
# but later for anyone who sat out, went undrafted and signed after a gap, or played in another
# league first.
COLLEGE_LAG_MIN = 1
COLLEGE_LAG_MAX = 5

# nflverse/ESPN spellings that differ from cfbfastR's team names beyond the generic rules below.
_SCHOOL_ALIASES = {
    "ole miss": "mississippi",
    "pitt": "pittsburgh",
    "usc": "southern california",
    "ucf": "central florida",
    "usf": "south florida",
    "smu": "southern methodist",
    "tcu": "texas christian",
    "byu": "brigham young",
    "lsu": "louisiana state",
    "miami fl": "miami",
    "miami fla": "miami",
    "miami oh": "miami ohio",
    "nc state": "north carolina state",
    "north carolina st": "north carolina state",
    "uconn": "connecticut",
    "umass": "massachusetts",
    "unlv": "nevada las vegas",
    "utep": "texas el paso",
    "utsa": "texas san antonio",
    "vt": "virginia tech",
    "cal": "california",
    "app state": "appalachian state",
}


def normalize_school(name) -> str:
    """Normalise a school name so the two sources' spellings compare equal.

    Handles the dominant pattern (``St.`` vs ``State``) generically and the irreducible ones
    (``Ole Miss`` -> ``Mississippi``) by table.
    """
    if name is None or name != name:
        return ""
    s = str(name).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = _SCHOOL_ALIASES.get(s, s)
    # "Michigan St" -> "Michigan State"; applied after aliases so table entries win.
    s = re.sub(r"\bst\b", "state", s)
    s = re.sub(r"\buniv(ersity)?\b", "", s).strip()
    return _SCHOOL_ALIASES.get(s, s)


def link_college_to_cohort(careers, college_careers, *, lag_min: int = COLLEGE_LAG_MIN,
                           lag_max: int = COLLEGE_LAG_MAX):
    """Attach each cohort QB's college career profile, refusing ambiguous matches.

    A candidate matches when normalised names are equal and the QB's **last college season** sits
    ``lag_min``-``lag_max`` years before NFL entry. Among survivors, preference order is:

    1. the school agrees with ``draft_college`` (decisive when we have it),
    2. otherwise the modal one-year gap between last college season and NFL entry,
    3. otherwise unmatched and flagged ``ambiguous`` — never a coin flip.

    Returns ``careers`` with college columns added plus ``college_matched``,
    ``college_match_reason`` and ``n_college_candidates``.
    """
    import numpy as np
    import pandas as pd

    col = college_careers.copy()
    col["_key"] = col["player"].map(normalize_name)
    col = col[col["_key"] != ""]
    col["_school"] = col["final_team"].map(normalize_school)

    car = careers.copy()
    car["_key"] = car["player_name"].map(normalize_name)
    car["_school"] = car["draft_college"].map(normalize_school) if "draft_college" in car else ""

    # Both frames describe careers, so names like `first_season` and `career_games` collide and
    # mean different things on each side. Prefix the college side rather than silently producing
    # duplicate columns whose provenance is ambiguous downstream.
    collisions = {c: f"college_{c}" for c in col.columns
                  if c in car.columns and c not in ("_key", "_school")}
    col = col.rename(columns=collisions)
    last_col = collisions.get("last_season", "last_season")

    feature_cols = [c for c in col.columns if c not in ("_key", "_school")]
    by_key = {k: g for k, g in col.groupby("_key")}

    picks, reasons, counts = [], [], []
    for _, row in car.iterrows():
        cands = by_key.get(row["_key"])
        if cands is None:
            picks.append(None)
            reasons.append("no_name_match")
            counts.append(0)
            continue

        entry = row["entry_season"]
        if entry != entry:
            picks.append(None)
            reasons.append("no_entry_season")
            counts.append(0)
            continue

        gap = int(entry) - cands[last_col]
        window = cands[(gap >= lag_min) & (gap <= lag_max)]
        counts.append(len(window))

        if len(window) == 0:
            picks.append(None)
            reasons.append("name_match_out_of_window")
            continue
        if len(window) == 1:
            picks.append(window.iloc[0])
            reasons.append("unique")
            continue

        # School is decisive when we have it, since two same-named QBs rarely share a program.
        same_school = window[window["_school"] == row["_school"]] if row["_school"] else window
        if len(same_school) == 1:
            picks.append(same_school.iloc[0])
            reasons.append("school_match")
            continue

        pool = same_school if len(same_school) else window
        best = pool.assign(_d=(int(entry) - pool[last_col] - 1).abs())
        top = best[best["_d"] == best["_d"].min()]
        if len(top) == 1:
            picks.append(top.iloc[0])
            reasons.append("modal_gap")
        else:
            picks.append(None)
            reasons.append("ambiguous")

    matched_frame = pd.DataFrame(
        [p[feature_cols] if p is not None else pd.Series(index=feature_cols, dtype="object")
         for p in picks],
        index=car.index,
    )
    out = pd.concat([car.drop(columns=["_key", "_school"]), matched_frame], axis=1)
    out["college_match_reason"] = reasons
    out["n_college_candidates"] = counts
    out["college_matched"] = np.array([p is not None for p in picks])
    return out


def college_coverage_report(linked, qb_seasons, college_careers) -> str:
    """Markdown report on college-layer coverage, join quality, and what the cohort looks like."""
    import pandas as pd

    label = "late_sustained"
    reach = linked[linked["entry_season"] >= 2005]

    def rate(df):
        n = len(df)
        return f"{int(df['college_matched'].sum())}/{n} ({100 * df['college_matched'].mean():.0f}%)"

    by_era = (
        linked.assign(entry_era=pd.cut(
            linked["entry_season"], bins=[1998, 2004, 2009, 2014, 2019, 2030],
            labels=["1999-2004", "2005-2009", "2010-2014", "2015-2019", "2020+"],
        ))
        .groupby("entry_era", observed=True)["college_matched"]
        .agg(n="size", matched="sum")
    )
    by_era["pct"] = (100 * by_era["matched"] / by_era["n"]).round(0)

    outcome = reach.copy()
    outcome["outcome"] = outcome[label].map({0.0: "on_time", 1.0: "late"}).fillna("never")
    by_outcome = outcome.groupby("outcome")["college_matched"].agg(n="size", matched="sum")
    by_outcome["pct"] = (100 * by_outcome["matched"] / by_outcome["n"]).round(0)

    reasons = linked["college_match_reason"].value_counts()
    matched_late = linked[(linked[label] == 1) & linked["college_matched"]]
    n_late_matched = len(matched_late)
    n_ever = int(((reach[label].notna()) & reach["college_matched"]).sum())
    trunc = linked["college_career_truncated"]
    n_truncated = int((trunc.fillna(0) == 1).sum())
    n_truncated_late = int((matched_late["college_career_truncated"].fillna(0) == 1).sum())

    def fmt(v):
        """Render a cell without pandas' float noise — counts as ints, rates to 3 places."""
        if v is None or v != v:
            return ""
        if isinstance(v, float):
            return str(int(v)) if v.is_integer() else f"{v:.3f}"
        return str(v)

    def table(df, index_name):
        head = f"| {index_name} | " + " | ".join(str(c) for c in df.columns) + " |"
        sep = "|" + "|".join("---" for _ in range(len(df.columns) + 1)) + "|"
        rows = [f"| {i} | " + " | ".join(fmt(v) for v in r) + " |"
                for i, r in zip(df.index, df.itertuples(index=False, name=None))]
        return "\n".join([head, sep, *rows])

    seasons_span = f"{int(qb_seasons['season'].min())}–{int(qb_seasons['season'].max())}"

    return f"""# College layer — coverage, join quality, and the late-breakout profile

The project's **primary** pre-NFL evidence. Built from cfbfastR play-by-play rather than a season
stats table so efficiency (EPA per dropback, success rate) is available and sacks separate cleanly
from rushing — NCAA box scores charge sack yardage against rushing, which badly understates mobile
QBs. Design: [`docs/QB_BREAKOUT_PLAN.md`](../docs/QB_BREAKOUT_PLAN.md) §2.2.

- **Play-by-play seasons aggregated:** {seasons_span}
- **QB-seasons:** {qb_seasons.height} (≥50 dropbacks) across {len(college_careers)} college careers
- **Cohort matched:** {rate(linked)} overall, {rate(reach)} for QBs entering 2005+

## Match rate by NFL entry era

Usable play-by-play starts in **2004** — 2002 and 2003 exist but ship an older, thinner schema
without `completion`, `pass_td` or `EPA_success`, so no QB-season can be built from them. QBs
entering the NFL before about 2005 therefore played their college careers off-camera entirely.

{table(by_era, "entry era")}

**Careers straddling the 2004 boundary are clipped, and flagged rather than hidden.** Aaron
Rodgers reads as one college season and 274 attempts because only 2004 is in range — not because
he was a one-year starter. `college_career_truncated` marks these ({n_truncated} of the matched
cohort, {n_truncated_late} of the late breakouts). Career totals are unusable for them; the
`final_*` block is not, since a final season is a final season either way. That is the argument
for leaning the feature set on final-season form.

## Match rate by outcome (2005+ entrants)

The check that matters: if late breakouts match at a materially lower rate than everyone else,
the modelling sample is biased against the cohort the project is about.

{table(by_outcome, "outcome")}

## Why rows failed to match

{table(reasons.to_frame("n"), "reason")}

## The late-breakout cohort, in college

{n_late_matched} late breakouts carry a college profile — against 6 for the high-school layer,
which is why the project went college-only.

{table(matched_late[["player_name", "entry_season", "final_team", "n_college_seasons",
                     "final_dropbacks", "final_epa_per_db", "final_success_rate",
                     "final_rush_share", "epa_trend", "transferred",
                     "college_career_truncated"]]
       .set_index("player_name"), "player")}

## What this supports

`ever_breakout` has **{n_ever}** matched QBs with a resolved outcome inside college reach — the
primary modelled outcome. Lateness, at {n_late_matched}, stays descriptive: archetype profiles and
base rates, not a classifier (`QB_BREAKOUT_PLAN.md` §5).
"""
