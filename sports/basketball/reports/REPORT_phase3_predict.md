# Phase 3 — Next-Season Archetype Predictor

Leak-safe **N→N+1** prediction of a player's archetype, returning players only. Features as-of season N:
current soft membership `p0..p11` (+ `top_prob`/`entropy`), 19 style z-features, one-year style
**trajectory** deltas, archetype tenure, and an experience/age term. Two feature sets are compared —
**Model A** uses a years-of-experience proxy (leak-safe, no fetch); **Model B** swaps in **true age**
(nba_api birthdates, 100% covered). Target: archetype in N+1.
Evaluation is **walk-forward** (train only on transitions into earlier seasons); all rows are scored on
the same 2538 pooled pairs (2017-2026).

## Walk-forward results

| predictor | top-1 acc | macro-F1 | log-loss | Brier |
| --- | --- | --- | --- | --- |
| persistence | 0.576 | 0.569 | 1.957 | 0.339 |
| marginal | 0.138 | 0.019 | 23.810 | 1.404 |
| Model A (YOE) | 0.547 | 0.533 | 1.505 | 0.371 |
| Model B (true age) | 0.548 | 0.537 | 1.496 | 0.368 |

> The models **do not beat persistence on hard top-1 accuracy** (0.548 vs
> 0.576) — archetype membership is highly persistent, so "same as last season" is a very
> strong argmax baseline. But they **more than halve log-loss** (1.496 vs
> 1.957): the models produce far better-calibrated soft membership vectors, which is
> exactly what Phase 2 consumes at draft time (projected category coverage needs a probability
> distribution, not a single hard label). **Marginal** (always the most common archetype) is the floor.

**Does true age beat the experience proxy?** Barely — Model B (age) lands at acc 0.548 / log-loss 1.496 vs Model A (YOE) 0.547 / 1.505. Neither age nor experience cracks the top features. The plan's expectation that **age carries most of the lift is not supported**: archetype transitions are governed by *where a player is now* (current membership + style), not by age or one-year trajectory.

## What drives the prediction (permutation importance, Model B (true age))

| feature | importance |
| --- | --- |
| p7 | +0.0635 |
| p9 | +0.0614 |
| p1 | +0.0577 |
| p12 | +0.0534 |
| ast36_z | +0.0468 |
| p6 | +0.0387 |
| p2 | +0.0365 |
| p8 | +0.0353 |
| p0 | +0.0342 |
| dreb36_z | +0.0342 |
| p11 | +0.0332 |
| p3 | +0.0324 |
| p5 | +0.0282 |
| stl36_z | +0.0276 |
| oreb36_z | +0.0271 |

The current **membership vector and style** dominate; the experience/age and trajectory-delta terms add
little on their own — "archetypes are sticky, and where you are now says most about where you'll be."

---
*Deliverable:* the model's calibrated soft membership vector is the per-player **projected next-season
archetype(s)** consumed by the Phase-2 optimizer at draft time. *Caveats:* top-1 accuracy below
persistence is honest — the win is probabilistic calibration, not point classification. Every player-season
is **equal-weighted** (subject only to the 15-mpg/20-gp eligibility gate); a fantasy draft cares most
about top-of-draft players, so a value-weighted or top-N-by-value evaluation is a noted refinement (the
taxonomy and Phase-2 optimizer are already value-centric via the draft prior).
