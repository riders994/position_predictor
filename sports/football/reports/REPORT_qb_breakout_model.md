# Predicting a QB breakout from college evidence alone

The constraint this project was built around is that the model may see **only pre-NFL evidence**.
This is the stage that finds out what that buys.

- **Sample:** 177 quarterbacks, **35 sustained
  breakouts** (base rate 0.198)
- **Outcome:** a top-15 PPR PPG season held at top-20 in ≥2 of the 3 seasons from it (§3.2)
- **Model:** L2 logistic regression, C=0.1, class-balanced — the most complex thing
  35 positives support

## Who is in the sample

A quarterback who entered in 2024 and has not broken out has not *failed* to break out. Labelling
him a negative teaches the model that recent profiles do not work. The sample is therefore
restricted to **5+ NFL seasons of opportunity**, on the evidence that 90% of
sustained breakouts happen by year 5. That threshold is reported rather than asserted:

| min NFL seasons | n | breakouts | base rate |
|---|---|---|---|
| 3 | 200 | 36 | 0.18 |
| 5 | 177 | 35 | 0.198 |
| 7 | 155 | 28 | 0.181 |
| 9 | 130 | 23 | 0.177 |
| 11 | 106 | 19 | 0.179 |

The base rate is flat from three seasons to eleven, so the cut is not doing hidden work — it
removes false negatives without reshaping the outcome.

## Does the college evidence predict anything at all?

| tier | features | CV AUC | ± sd | P@15 | Brier | temporal AUC |
|---|---|---|---|---|---|---|
| portable | 12 | 0.696 | 0.018 | 0.51 | 0.211 | 0.725 |
| full | 19 | 0.689 | 0.021 | 0.477 | 0.211 | 0.713 |

**Yes, decisively — against chance.** Shuffling the labels and re-running the whole pipeline
200 times gives a null AUC of **0.493 ± 0.071** (95th
percentile 0.606). The observed 0.696 sits far outside
it, **p = 0.0050**. College production is not noise.

The temporal split — train on early entrants, score later ones, which is the only evaluation whose
information flow matches real use — holds up at **0.725**
(78 test quarterbacks, 17 breakouts).

## What portability costs: nothing

This is the question stage 4 set up, and the answer is clean.

The **portable** tier uses only features CFBD also computes, so it can be pointed at a quarterback
whose college career ended last autumn. The **full** tier adds EPA per dropback, success rate and
the career-shape features derived from them — better measurements, available 2004–2021 only.

The portable model scores **0.696** against the full model's
**0.689**: a gap of +0.007, well inside the fold-to-fold spread of either
(± 0.018). **The efficiency features add nothing detectable.** The project
therefore keeps the forward-looking tool at no measured cost — and the reason is visible in the
coefficients below: what the model is actually reading is workload and role, not efficiency.

Gradient boosting was fitted too and does not help
(0.679), which is the expected result at this N and is reported so that
nobody has to wonder whether a more flexible learner was tried.

## What it reads

| feature | coef | sign_stability |
|---|---|---|
| final_attempts | 0.482 | 1 |
| archetype_name_pocket_quick | -0.364 | 1 |
| final_rush_share | 0.36 | 1 |
| archetype_stability | 0.298 | 0.98 |
| final_yards_per_attempt | 0.245 | 1 |
| archetype_changed | 0.189 | 0.8 |
| final_td_rate | 0.186 | 0.9 |
| final_yards_per_completion | 0.169 | 0.95 |
| transferred | -0.149 | 0.9 |
| final_completion_pct | 0.087 | 0.75 |

`sign_stability` is the share of bootstrap refits keeping the sign. **Volume is the strongest
single input** — final-season attempts — followed by *not* being a quick-game pocket passer and by
rushing share. That ordering is worth sitting with: the model's best evidence is how much a
quarterback played and what role he had, not how well he threw.

## The draft, as an independent opinion

Draft position is never a feature here — it is the league's own verdict, and mixing it in would
make the model partly a report of what scouts already decided. Kept separate, it is a benchmark:

| Ranking | AUC | Precision@15 |
|---|---|---|
| College model (portable) | 0.696 | 0.510 |
| **Draft position** | **0.890** | **0.733** |

**The draft is far better, and that comparison is not what it looks like.**

| draft_band | n | breakouts | rate | mean_nfl_games | mean_nfl_starts |
|---|---|---|---|---|---|
| R1 | 46 | 26 | 0.565 | 84.565 | 79.935 |
| R2-3 | 43 | 7 | 0.163 | 45.791 | 38.605 |
| R4+/UDFA | 88 | 2 | 0.023 | 17.875 | 11.943 |

A fantasy breakout requires snaps, and the draft *allocates* snaps: first-rounders average
85 career games against
18 for day-three and undrafted quarterbacks. The draft
partly **causes** the outcome it appears to predict. Beating it is not the standard a college-only
model should be held to, and a model that *did* beat it would be suspect.

The fair question is whether college evidence separates breakouts among **similarly drafted**
quarterbacks — which is also the late-breakout question, since a late breakout is by construction
someone the league undervalued:

| draft_band | n | breakouts | model_auc | draft_auc |
|---|---|---|---|---|
| R1 | 46 | 26 | 0.496 | 0.661 |
| R2-3 | 43 | 7 | 0.675 | 0.651 |
| R4+/UDFA | 88 | 2 | 0.686 | 0.814 |

**And here the answer is no, where it would matter most.** Among first-round picks the college
model is at chance (0.496, n=46) — once the league has decided a
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
   35 breakouts here, 9 were late — enough to describe,
   not enough to fit. Stage 5 reached the same wall from a different direction.

The negative result is worth as much as a positive one would have been: it says that if late
breakouts are visible before the NFL, they are not visible in college *production*. What remains
untested is context — competition faced, supporting cast, scheme, the conditions a quarterback
produced under rather than the totals he produced. That is where a follow-on would go.
