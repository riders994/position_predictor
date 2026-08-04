# Drafting situation: franchise and regime

Does *where* a quarterback landed matter beyond *how highly* he was picked? The question has been
open since the project's first prompt — "player archetypes, drafting situations, and other
factors" — and this stage answers it as far as the arithmetic allows, which is not very far.

## Why there are no team breakout rates in this report

140 drafted quarterbacks with a settled outcome, spread across 32 franchises: a **median
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

### Drafting franchise

| draft_franchise | qbs | observed | expected | diff | p_two_sided |
|---|---|---|---|---|---|
| BAL | 5 | 2 | 0.44 | 1.56 | 0.058 |
| WAS | 6 | 3 | 1.57 | 1.43 | 0.227 |
| PHI | 5 | 2 | 0.67 | 1.33 | 0.133 |
| GB | 5 | 2 | 0.78 | 1.22 | 0.169 |
| SF | 4 | 2 | 1.2 | 0.8 | 0.36 |
| LA | 4 | 1 | 1.86 | -0.86 | 0.357 |
| NE | 6 | 0 | 0.95 | -0.95 | 0.574 |
| CLE | 7 | 1 | 2.14 | -1.14 | 0.38 |
| LV | 4 | 0 | 1.27 | -1.27 | 0.138 |
| TEN | 6 | 0 | 2.18 | -2.18 | 0.031 |

*(best and worst of 28 regimes with ≥3 quarterbacks; `diff` is
breakouts above what their draft picks predicted.)*

**Individually 1 regime clears p < 0.05, against 1.4 expected by chance from 28 uncorrected comparisons.** Across all 28 evaluated, the spread of observed-minus-expected is **p = 0.128** against random reassignment — **exactly what chance produces.** There is no franchise or regime effect visible here once draft capital is accounted for.

**What would have been detectable.** A typical regime here drafted
4 quarterbacks, expecting 0.98 breakouts.
To clear a 5% threshold it would have needed **3** — about
2.02 extra breakouts above expectation from
4 quarterbacks. Nothing subtler than that could have been found,
whether or not it is there.

### Head coach at the draft

| draft_coach | qbs | observed | expected | diff | p_two_sided |
|---|---|---|---|---|---|
| Mike Shanahan | 3 | 3 | 1.29 | 1.71 | 0.026 |
| John Harbaugh | 3 | 2 | 0.34 | 1.66 | 0.027 |
| Andy Reid | 5 | 2 | 0.95 | 1.05 | 0.222 |
| Marvin Lewis | 3 | 1 | 0.33 | 0.67 | 0.296 |
| Sean McDermott | 3 | 1 | 0.67 | 0.33 | 1 |
| Doug Marrone | 3 | 0 | 0.49 | -0.49 | 1 |
| Tony Sparano | 3 | 0 | 0.51 | -0.51 | 0.645 |
| Jeff Fisher | 5 | 1 | 1.77 | -0.77 | 0.471 |
| Ken Whisenhunt | 3 | 0 | 0.9 | -0.9 | 0.298 |
| Bill Belichick | 6 | 0 | 0.95 | -0.95 | 0.575 |

*(best and worst of 15 regimes with ≥3 quarterbacks; `diff` is
breakouts above what their draft picks predicted.)*

**Individually 2 regimes clear p < 0.05, against 0.8 expected by chance from 15 uncorrected comparisons.** Across all 15 evaluated, the spread of observed-minus-expected is **p = 0.089** against random reassignment — **exactly what chance produces.** There is no franchise or regime effect visible here once draft capital is accounted for.

**What would have been detectable.** A typical regime here drafted
3 quarterbacks, expecting 0.73 breakouts.
To clear a 5% threshold it would have needed **3** — about
2.27 extra breakouts above expectation from
3 quarterbacks. Nothing subtler than that could have been found,
whether or not it is there.

## General managers

**Not analysed, because there is no data for it.** nflverse publishes head coaches (via schedules, 1999+) but not general managers, and no free structured GM-by-team-season table exists. Head coach is used above as the available regime proxy — an imperfect one, since a coach and a general manager can disagree about a quarterback and often do.

`situation/regime.py::attach_gm` takes a hand-supplied `(season, team, gm)` table and the script accepts `--gm-table`, so this is a one-CSV job whenever a list is available. Given the power arithmetic above it would not change the conclusion: general-manager tenures are *shorter* than franchise histories, so the cells would be smaller still.

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
