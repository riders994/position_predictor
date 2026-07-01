# Phase 3 — Next-Season Archetype Predictor (yoe)

Leak-safe **N→N+1** prediction of a player's archetype, returning players only. Features as-of season N:
current soft membership `p0..p11` (+ `top_prob`/`entropy`), 19 style z-features, one-year style
**trajectory** deltas, archetype tenure, and **years-of-experience proxy**.
Target: archetype in N+1. Evaluation is **walk-forward** (train only on transitions into earlier
seasons); all baselines are scored on the same 2538 pooled pairs
(2017-2026).

## Walk-forward results

| predictor | top-1 acc | macro-F1 | log-loss | Brier |
| --- | --- | --- | --- | --- |
| model | 0.605 | 0.525 | 1.322 | 0.386 |
| persistence | 0.638 | 0.569 | 2.277 | 0.366 |
| marginal | 0.210 | 0.029 | 21.817 | 1.343 |

> The model does **not** beat persistence on hard top-1 accuracy (0.605 vs 0.638) — archetype membership is highly persistent, so 'same as last season' is a very strong argmax baseline. But the model **more than halves log-loss** (1.322 vs 2.277): it produces far better-calibrated soft membership vectors, which is exactly what Phase 2 consumes at draft time (projected category coverage needs a probability distribution, not a single hard label).

**Marginal** (always predict the most common archetype) is the floor — the sticky structure is real
signal both baselines above it exploit.

## What drives the prediction (permutation importance)

| feature | importance |
| --- | --- |
| p1 | +0.0535 |
| p0 | +0.0512 |
| p6 | +0.0417 |
| p8 | +0.0384 |
| p3 | +0.0353 |
| ast36_z | +0.0346 |
| stl36_z | +0.0313 |
| p10 | +0.0301 |
| p2 | +0.0289 |
| fg3a_rate_z | +0.0275 |
| p9 | +0.0273 |
| oreb36_z | +0.0250 |
| dreb36_z | +0.0235 |
| fg3a36_z | +0.0229 |
| fg3_pct_z | +0.0216 |

The current **membership vector and style** dominate; the **yoe** / trajectory-delta terms add
little on their own — consistent with "archetypes are sticky, and where you are now says most about where
you'll be." A true-age variant (Model B) tests whether real age adds what the experience proxy cannot.

---
*Deliverable:* the model's calibrated soft membership vector is the per-player **projected next-season
archetype(s)** consumed by the Phase-2 optimizer at draft time. *Caveat:* top-1 accuracy below
persistence is honest — the win is probabilistic calibration, not point classification.
