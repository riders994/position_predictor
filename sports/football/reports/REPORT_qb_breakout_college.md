# College layer — coverage, join quality, and the late-breakout profile

The project's **primary** pre-NFL evidence. Built from cfbfastR play-by-play rather than a season
stats table so efficiency (EPA per dropback, success rate) is available and sacks separate cleanly
from rushing — NCAA box scores charge sack yardage against rushing, which badly understates mobile
QBs. Design: [`docs/QB_BREAKOUT_PLAN.md`](../docs/QB_BREAKOUT_PLAN.md) §2.2.

- **Play-by-play seasons aggregated:** 2004–2021
- **QB-seasons:** 3419 (≥50 dropbacks) across 1684 college careers
- **Cohort matched:** 215/331 (65%) overall, 215/244 (88%) for QBs entering 2005+

## Match rate by NFL entry era

Usable play-by-play starts in **2004** — 2002 and 2003 exist but ship an older, thinner schema
without `completion`, `pass_td` or `EPA_success`, so no QB-season can be built from them. QBs
entering the NFL before about 2005 therefore played their college careers off-camera entirely.

| entry era | n | matched | pct |
|---|---|---|---|
| 1999-2004 | 87 | 0 | 0 |
| 2005-2009 | 54 | 44 | 81 |
| 2010-2014 | 61 | 56 | 92 |
| 2015-2019 | 62 | 57 | 92 |
| 2020+ | 67 | 58 | 87 |

**Careers straddling the 2004 boundary are clipped, and flagged rather than hidden.** Aaron
Rodgers reads as one college season and 274 attempts because only 2004 is in range — not because
he was a one-year starter. `college_career_truncated` marks these (30 of the matched
cohort, 2 of the late breakouts). Career totals are unusable for them; the
`final_*` block is not, since a final season is a final season either way. That is the argument
for leaning the feature set on final-season form.

## Match rate by outcome (2005+ entrants)

The check that matters: if late breakouts match at a materially lower rate than everyone else,
the modelling sample is biased against the cohort the project is about.

| outcome | n | matched | pct |
|---|---|---|---|
| late | 11 | 9 | 82 |
| never | 202 | 177 | 88 |
| on_time | 31 | 29 | 94 |

## Why rows failed to match

| reason | n |
|---|---|
| unique | 210 |
| no_name_match | 116 |
| modal_gap | 4 |
| school_match | 1 |

## The late-breakout cohort, in college

9 late breakouts carry a college profile — against 6 for the high-school layer,
which is why the project went college-only.

| player | entry_season | final_team | n_college_seasons | final_dropbacks | final_epa_per_db | final_success_rate | final_rush_share | epa_trend | transferred | college_career_truncated |
|---|---|---|---|---|---|---|---|---|---|---|
| Aaron Rodgers | 2005 | California | 1 | 286 | 0.408 | 0.570 | 0.176 | 0 | 0 | 1 |
| Alex Smith | 2005 | Utah | 1 | 244 | 0.572 | 0.615 | 0.313 | 0 | 0 | 1 |
| Tyrod Taylor | 2011 | Virginia Tech | 4 | 295 | 0.166 | 0.444 | 0.257 | 0.266 | 0 | 0 |
| Kirk Cousins | 2012 | Michigan State | 3 | 391 | 0.323 | 0.537 | 0.049 | -0.008 | 0 | 0 |
| Ryan Tannehill | 2012 | Texas A&M | 2 | 475 | 0.257 | 0.528 | 0.088 | 0.040 | 0 | 0 |
| Geno Smith | 2013 | West Virginia | 3 | 491 | 0.278 | 0.556 | 0.084 | 0.161 | 0 | 0 |
| Jameis Winston | 2015 | Florida State | 2 | 461 | 0.191 | 0.551 | 0.080 | -0.492 | 0 | 0 |
| Baker Mayfield | 2018 | Oklahoma | 4 | 407 | 0.466 | 0.585 | 0.149 | 0.160 | 1 | 0 |
| Jordan Love | 2020 | Utah State | 3 | 464 | 0.025 | 0.444 | 0.116 | 0.064 | 0 | 0 |

## What this supports

`ever_breakout` has **38** matched QBs with a resolved outcome inside college reach — the
primary modelled outcome. Lateness, at 9, stays descriptive: archetype profiles and
base rates, not a classifier (`QB_BREAKOUT_PLAN.md` §5).
