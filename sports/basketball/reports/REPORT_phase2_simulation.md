# Phase 2 — Simulation & Representation Shootout

**Why simulate.** The real-team composition model hit a four-way null: `cat_win_rate` on real rosters is
dominated by in-season management, streaming, injuries and schedule luck, which swamp the
draft-composition signal. The simulator isolates the draft-composition question — it drafts teams from a
**leak-safe prior-season-value** prior with **punt-strategy variation** and a ~30% auto-draft / 70%
manager mix, then scores each team by round-robin **9-cat** head-to-head from the players' *actual*
season production.

**Corpus:** 8640 simulated team-seasons (60 leagues × 12 seasons,
2015-2026; returning players only, prior-season prior). Seed 1729.

## Representation shootout — which features carry the signal

Leave-one-season-out Ridge, target `sim_cat_win_rate` (league mean 0.5 by construction):

| representation | out-of-fold R² | out-of-fold MAE | # features |
| --- | --- | --- | --- |
| archetype shares | +0.0523 | 0.0608 | 13 |
| prior coverage | +0.0436 | 0.0610 | 9 |
| actual coverage | +0.3695 | 0.0495 | 9 |

**Three conclusions:**

1. **Archetype shares are the wrong success representation** (R²=+0.052 — null even in this
   clean sim). They describe play-style but *discard the category information* that decides 9-cat
   matchups; a contribution-matrix can't rescue a linear model since Ridge already spans any linear
   transform of shares.
2. **Category coverage is the mechanism** (actual coverage R²=+0.369). The team's realized
   9-cat z-profile largely *determines* which categories it wins — so success should be modeled on
   coverage, with archetypes kept as the interpretable taxonomy.
3. **Draft-time projection is the binding constraint** (prior coverage R²=+0.044). Last
   season's profile carries only a little of next season's coverage because players don't reproduce
   their z-profile — so the ceiling on a *draftable* model is projecting next-season production.

## Draft-strategy leaderboard — the signal lives in the punt builds

Mean `sim_cat_win_rate` by drafter strategy, with z vs the 0.5 league mean. That some punts clear 0.5 by
many standard errors (and others fall below) is the positive control: category coverage, built
deliberately, wins.

| strategy | n | mean cat-win-rate | z vs 0.5 |
| --- | --- | --- | --- |
| punt_ft | 460 | 0.5414 | +10.1 |
| punt_pts | 495 | 0.5102 | +3.1 |
| punt_ft_ast | 477 | 0.5101 | +3.0 |
| balanced | 3078 | 0.5068 | +4.4 |
| punt_stl | 454 | 0.5052 | +1.8 |
| punt_fg | 452 | 0.4987 | -0.3 |
| punt_blk | 455 | 0.4978 | -0.6 |
| punt_reb | 457 | 0.4953 | -1.4 |
| punt_fg3m | 479 | 0.4947 | -2.4 |
| punt_ast | 481 | 0.4936 | -2.1 |
| punt_fg3m_ft | 440 | 0.4918 | -3.8 |
| punt_blk_stl | 468 | 0.4589 | -13.4 |
| punt_tov | 444 | 0.4552 | -9.8 |

---
*Honesty:* a fully-simulated model partly bakes in the archetype→category mapping, so these are
*directional* findings, reality-checked against the real Yahoo/Fantrax team-seasons. The leak-safe
prior-season prior is the realism floor; real mock/ADP draft order is the v2 upgrade. The actionable
follow-on is a **roster optimizer** that targets category coverage for a chosen punt build (see
`make optimize`).
