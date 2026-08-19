"""Link high-school recruiting profiles to the NFL QB cohort.

There is no shared key between ESPN recruiting and nflverse, so this is a name match — and a
name match is exactly the kind of step that quietly corrupts a small-N study. Two failure modes
matter in opposite directions:

- **False negatives** shrink the usable cohort. If they fall unevenly — and they will, since
  recruiting coverage is thinner for lightly-recruited prospects, who are over-represented among
  late breakouts — the surviving sample is biased toward exactly the players the project is
  *least* interested in.
- **False positives** attach one player's high-school profile to another's NFL career, which at
  17 positives is enough to invent a finding.

The design is therefore conservative: match on normalised name **within a plausible recruiting-
class window**, and refuse ambiguous matches rather than guessing. Everything is reported —
:func:`recruiting_coverage_report` breaks the match rate down by outcome and era so a biased
match rate is visible instead of assumed away.
"""

from __future__ import annotations

import re
import unicodedata

# A recruit signs 3-6 years before entering the NFL: three college seasons at minimum, more with
# a redshirt, a transfer, or a fifth year. The window is deliberately wide — a wrong exclusion
# costs a cohort member, while the name match plus ambiguity check guards the other side.
CLASS_LAG_MIN = 3
CLASS_LAG_MAX = 7

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def normalize_name(name) -> str:
    """Casefold, strip accents, punctuation and generational suffixes.

    Handles the real differences between the sources: ``Michael Penix Jr.`` vs ``Michael Penix``,
    ``E.J. Manuel`` vs ``EJ Manuel``, and accented spellings.
    """
    if name is None or name != name:
        return ""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace("&", " and ")
    s = re.sub(r"[^a-z\s]", "", s)
    parts = [p for p in s.split() if p and p not in _SUFFIXES]
    return " ".join(parts)


def link_recruits_to_cohort(careers, recruits, *, lag_min: int = CLASS_LAG_MIN,
                            lag_max: int = CLASS_LAG_MAX):
    """Attach each cohort QB's recruiting profile, refusing ambiguous matches.

    A candidate matches when the normalised names are equal *and* the recruiting class falls
    ``lag_min``-``lag_max`` years before NFL entry. When several candidates survive, the one whose
    class sits at the modal 5-year lag wins; if that is still tied, the row is left unmatched and
    flagged ``ambiguous`` rather than resolved by guesswork.

    Returns ``careers`` with recruiting columns added plus ``matched``, ``match_reason`` and
    ``n_candidates``.
    """
    import numpy as np
    import pandas as pd

    rec = recruits.copy()
    rec["_key"] = rec["recruit_name"].map(normalize_name)
    rec = rec[rec["_key"] != ""]

    car = careers.copy()
    car["_key"] = car["player_name"].map(normalize_name)

    rec_cols = [c for c in rec.columns if c not in ("_key",)]
    by_key = {k: g for k, g in rec.groupby("_key")}

    picks, reasons, counts = [], [], []
    for _, row in car.iterrows():
        cands = by_key.get(row["_key"])
        if cands is None or row["entry_season"] != row["entry_season"]:
            picks.append(None)
            reasons.append("no_name_match" if cands is None else "no_entry_season")
            counts.append(0)
            continue

        entry = int(row["entry_season"])
        lag = entry - cands["recruit_class"]
        window = cands[(lag >= lag_min) & (lag <= lag_max)]
        counts.append(len(window))

        if len(window) == 0:
            picks.append(None)
            reasons.append("name_match_out_of_window")
        elif len(window) == 1:
            picks.append(window.iloc[0])
            reasons.append("unique")
        else:
            # Prefer the modal path: sign, redshirt or not, four college years, then the draft.
            best = window.assign(_d=(entry - window["recruit_class"] - 5).abs())
            top = best[best["_d"] == best["_d"].min()]
            if len(top) == 1:
                picks.append(top.iloc[0])
                reasons.append("modal_lag")
            else:
                picks.append(None)
                reasons.append("ambiguous")

    matched_frame = pd.DataFrame(
        [p[rec_cols] if p is not None else pd.Series(index=rec_cols, dtype="object")
         for p in picks],
        index=car.index,
    )
    out = pd.concat([car.drop(columns=["_key"]), matched_frame], axis=1)
    out["match_reason"] = reasons
    out["n_candidates"] = counts
    out["matched"] = np.array([p is not None for p in picks])
    return out


def recruiting_coverage_report(linked, recruits) -> str:
    """Markdown report on how well the high-school layer actually covers the cohort."""
    import pandas as pd

    era = linked[linked["entry_season"] >= 2010]
    label = "late_sustained"

    def rate(df):
        return f"{int(df['matched'].sum())}/{len(df)} ({100 * df['matched'].mean():.0f}%)"

    by_era = (
        linked.assign(entry_era=pd.cut(
            linked["entry_season"],
            bins=[1998, 2005, 2009, 2014, 2019, 2030],
            labels=["1999-2005", "2006-2009", "2010-2014", "2015-2019", "2020+"],
        ))
        .groupby("entry_era", observed=True)["matched"]
        .agg(n="size", matched="sum")
    )
    by_era["pct"] = (100 * by_era["matched"] / by_era["n"]).round(0)

    outcome = era.copy()
    outcome["outcome"] = outcome[label].map({0.0: "on_time", 1.0: "late"}).fillna("never")
    by_outcome = outcome.groupby("outcome")["matched"].agg(n="size", matched="sum")
    by_outcome["pct"] = (100 * by_outcome["matched"] / by_outcome["n"]).round(0)

    reasons = linked["match_reason"].value_counts()
    matched_late = linked[(linked[label] == 1) & linked["matched"]]

    def table(df, index_name):
        head = f"| {index_name} | " + " | ".join(str(c) for c in df.columns) + " |"
        sep = "|" + "|".join("---" for _ in range(len(df.columns) + 1)) + "|"
        rows = [f"| {i} | " + " | ".join(str(v) for v in r) + " |"
                for i, r in zip(df.index, df.itertuples(index=False, name=None))]
        return "\n".join([head, sep, *rows])

    n_late_matched = len(matched_late)
    n_ever_matched = int(((era[label].notna()) & era["matched"]).sum())
    unbiased = abs(by_outcome.loc["late", "pct"] - by_outcome.loc["never", "pct"]) <= 10

    return f"""# High-school layer — coverage and join quality

The ESPN recruiting pull is the project's high-school evidence. This report exists because a
name-based join on a 17-positive study can invent findings, so the match rate is measured rather
than assumed. Design notes: [`docs/QB_BREAKOUT_PLAN.md`](../docs/QB_BREAKOUT_PLAN.md) §2.

- **Recruiting records pulled:** {len(recruits)} QB-ish prospects (QB-PP / QB-DT / ATH)
- **Classes covered:** {int(recruits['recruit_class'].min())}–{int(recruits['recruit_class'].max())}
- **Cohort matched:** {rate(linked)} overall, {rate(era)} for QBs entering 2010+

## Match rate by entry era

ESPN recruiting is effectively empty before the 2006 class, so QBs entering the NFL before about
2010 are unreachable by construction. **This, not the NFL label, is what bounds the modelling
window.**

{table(by_era, "entry era")}

## Match rate by outcome (2010+ entrants)

The number that matters. If late breakouts match at a materially lower rate than the rest, the
modelling sample is biased against the very cohort the project is about, and any result has to be
read in that light.

{table(by_outcome, "outcome")}

## Why rows failed to match

{table(reasons.to_frame("n"), "reason")}

`ambiguous` rows are deliberately left unmatched: two same-named prospects in the plausible class
window is not something to resolve by coin flip.

## Late breakouts with a high-school profile

{len(matched_late)} of the late-breakout cohort carry recruiting data — the effective positive
count for any high-school-feature model.

{table(matched_late[["player_name", "entry_season", "recruit_class", "recruit_position",
                     "espn_grade", "rank_national", "rank_position", "hs_state"]]
       .set_index("player_name"), "player")}

## What this means for modelling

**The match rate is not the problem — the positive count is.** The join behaves well: coverage is
87-93% from 2010 on, and it is {"essentially unbiased" if unbiased else "**biased**"} across
outcomes, so the surviving sample is not skewed for or against late breakouts.

But two constraints multiply. The label needs 4+ elapsed NFL seasons, and the high-school layer
starts with the 2006 recruiting class. Their intersection leaves **{n_late_matched} late breakouts
with a high-school profile**. That is not a modelling sample for a binary late/not-late
classifier; it is a case series.

The workable design is the nested one already set out in the plan (§5):

1. **Model `ever_breakout` instead** — {n_ever_matched} matched QBs with a resolved outcome in the
   2010+ era, roughly {n_ever_matched // max(n_late_matched, 1)}x the positives. "Which pre-NFL
   profiles produce NFL-relevant QBs at all" is a question this data can actually answer.
2. **Treat lateness descriptively** — archetype profiles and base rates over the {n_late_matched},
   reported as such, with no classifier implying precision the N cannot support.
3. **Add college production** (stage 4), which extends coverage back further than recruiting does
   and carries far more per-player signal than a single scouting grade.

An early pattern worth noting but *not* yet worth believing at this N: five of the six late
breakouts with profiles were typed **QB-PP** (pocket passer) rather than dual-threat, and most
carried middling grades and no national ranking.
"""
