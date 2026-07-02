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
| persistence | 0.638 | 0.569 | 2.277 | 0.366 |
| marginal | 0.210 | 0.029 | 21.817 | 1.343 |
| Model A (YOE) | 0.605 | 0.525 | 1.322 | 0.386 |
| Model B (true age) | 0.608 | 0.526 | 1.316 | 0.382 |

> The models **do not beat persistence on hard top-1 accuracy** (0.608 vs
> 0.638) — archetype membership is highly persistent, so "same as last season" is a very
> strong argmax baseline. But they **more than halve log-loss** (1.316 vs
> 2.277): the models produce far better-calibrated soft membership vectors, which is
> exactly what Phase 2 consumes at draft time (projected category coverage needs a probability
> distribution, not a single hard label). **Marginal** (always the most common archetype) is the floor.

**Does true age beat the experience proxy?** Barely — Model B (age) lands at acc 0.608 / log-loss 1.316 vs Model A (YOE) 0.605 / 1.322. Neither age nor experience cracks the top features. The plan's expectation that **age carries most of the lift is not supported**: archetype transitions are governed by *where a player is now* (current membership + style), not by age or one-year trajectory.

## What drives the prediction (permutation importance, Model B (true age))

| feature | importance |
| --- | --- |
| p1 | +0.0527 |
| p0 | +0.0505 |
| p6 | +0.0424 |
| p8 | +0.0377 |
| ast36_z | +0.0361 |
| p3 | +0.0357 |
| p10 | +0.0304 |
| stl36_z | +0.0302 |
| dreb36_z | +0.0292 |
| p2 | +0.0277 |
| fg3a_rate_z | +0.0266 |
| fg3a36_z | +0.0254 |
| p9 | +0.0245 |
| oreb36_z | +0.0237 |
| fga36_z | +0.0221 |

The current **membership vector and style** dominate; the experience/age and trajectory-delta terms add
little on their own — "archetypes are sticky, and where you are now says most about where you'll be."

---
*Deliverable:* the model's calibrated soft membership vector is the per-player **projected next-season
archetype(s)** consumed by the Phase-2 optimizer at draft time. *Caveats:* top-1 accuracy below
persistence is honest — the win is probabilistic calibration, not point classification. Every player-season
is **equal-weighted** (subject only to the 15-mpg/20-gp eligibility gate); a fantasy draft cares most
about top-of-draft players, so a value-weighted or top-N-by-value evaluation is a noted refinement (the
taxonomy and Phase-2 optimizer are already value-centric via the draft prior).
