# High-school layer — coverage and join quality

The ESPN recruiting pull is the project's high-school evidence. This report exists because a
name-based join on a 17-positive study can invent findings, so the match rate is measured rather
than assumed. Design notes: [`docs/QB_BREAKOUT_PLAN.md`](../docs/QB_BREAKOUT_PLAN.md) §2.

- **Recruiting records pulled:** 22175 QB-ish prospects (QB-PP / QB-DT / ATH)
- **Classes covered:** 2006–2026
- **Cohort matched:** 168/331 (51%) overall, 166/190 (87%) for QBs entering 2010+

## Match rate by entry era

ESPN recruiting is effectively empty before the 2006 class, so QBs entering the NFL before about
2010 are unreachable by construction. **This, not the NFL label, is what bounds the modelling
window.**

| entry era | n | matched | pct |
|---|---|---|---|
| 1999-2005 | 99 | 0 | 0.0 |
| 2006-2009 | 42 | 2 | 5.0 |
| 2010-2014 | 61 | 47 | 77.0 |
| 2015-2019 | 62 | 57 | 92.0 |
| 2020+ | 67 | 62 | 93.0 |

## Match rate by outcome (2010+ entrants)

The number that matters. If late breakouts match at a materially lower rate than the rest, the
modelling sample is biased against the very cohort the project is about, and any result has to be
read in that light.

| outcome | n | matched | pct |
|---|---|---|---|
| late | 7 | 6 | 86.0 |
| never | 150 | 130 | 87.0 |
| on_time | 33 | 30 | 91.0 |

## Why rows failed to match

| reason | n |
|---|---|
| unique | 156 |
| no_name_match | 150 |
| name_match_out_of_window | 12 |
| modal_lag | 12 |
| ambiguous | 1 |

`ambiguous` rows are deliberately left unmatched: two same-named prospects in the plausible class
window is not something to resolve by coin flip.

## Late breakouts with a high-school profile

6 of the late-breakout cohort carry recruiting data — the effective positive
count for any high-school-feature model.

| player | entry_season | recruit_class | recruit_position | espn_grade | rank_national | rank_position | hs_state |
|---|---|---|---|---|---|---|---|
| Tyrod Taylor | 2011 | 2007.0 | QB-PP | 84.0 | 16.0 | 3.0 | VA |
| Kirk Cousins | 2012 | 2007.0 | QB-PP | 70.0 | nan | 137.0 | MI |
| Ryan Tannehill | 2012 | 2007.0 | QB-PP | 77.0 | nan | 34.0 | TX |
| Jimmy Garoppolo | 2014 | 2010.0 | QB-PP | 40.0 | nan | nan | IL |
| Baker Mayfield | 2018 | 2013.0 | QB-PP | 70.0 | nan | 69.0 | TX |
| Jordan Love | 2020 | 2016.0 | QB-DT | 70.0 | nan | 62.0 | CA |

## What this means for modelling

**The match rate is not the problem — the positive count is.** The join behaves well: coverage is
87-93% from 2010 on, and it is essentially unbiased across
outcomes, so the surviving sample is not skewed for or against late breakouts.

But two constraints multiply. The label needs 4+ elapsed NFL seasons, and the high-school layer
starts with the 2006 recruiting class. Their intersection leaves **6 late breakouts
with a high-school profile**. That is not a modelling sample for a binary late/not-late
classifier; it is a case series.

The workable design is the nested one already set out in the plan (§5):

1. **Model `ever_breakout` instead** — 36 matched QBs with a resolved outcome in the
   2010+ era, roughly 6x the positives. "Which pre-NFL
   profiles produce NFL-relevant QBs at all" is a question this data can actually answer.
2. **Treat lateness descriptively** — archetype profiles and base rates over the 6,
   reported as such, with no classifier implying precision the N cannot support.
3. **Add college production** (stage 4), which extends coverage back further than recruiting does
   and carries far more per-player signal than a single scouting grade.

An early pattern worth noting but *not* yet worth believing at this N: five of the six late
breakouts with profiles were typed **QB-PP** (pocket passer) rather than dual-threat, and most
carried middling grades and no national ranking.
