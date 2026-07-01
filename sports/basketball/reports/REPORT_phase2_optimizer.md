# Phase 2 — Roster Optimizer (category-coverage builds)

The optimizer drafts the roster that maximizes **projected category coverage** for a chosen build. It
scores each candidate by the field-calibrated win map — modeling the field's per-category team coverage
as `N(mu_c, sigma_c)`, a roster's value is `mean_c Phi((cov_c - mu_c)/sigma_c)` over the *contested*
categories — using players' **prior-season** z-profiles (the draft-time projection). The Gaussian-CDF
shape makes it punt-aware: it stops piling onto locked categories and shores up winnable ones.

## In-sim validation

Each build seats one optimizer team against the manager/auto field (25 leagues ×
12 seasons) and records its realized `cat_win_rate`. `top-of-field` is the share of leagues
the optimizer finishes best of 12 — **random ≈ 0.083**.

| build | optimizer | field | lift | top-of-field |
| --- | --- | --- | --- | --- |
| punt_ft | 0.5370 | 0.4966 | +0.0404 | 0.217 |
| punt_fg3m_ft | 0.5223 | 0.4980 | +0.0244 | 0.087 |
| punt_pts | 0.5209 | 0.4981 | +0.0228 | 0.150 |
| balanced | 0.5089 | 0.4992 | +0.0097 | 0.093 |
| punt_ft_ast | 0.5019 | 0.4998 | +0.0021 | 0.090 |
| punt_fg | 0.4989 | 0.5001 | -0.0012 | 0.100 |
| punt_tov | 0.4680 | 0.5029 | -0.0349 | 0.043 |

The optimizer beats the field, and **most on punt builds** — `punt_ft` is the strongest, consistent with
the 9-cat meta (concede one low-leverage percentage category to dominate the rest). Punting a
cheap-to-win category (`punt_tov`) backfires. The balanced lift is small by design: the shootout showed
draft-time projection is the binding constraint, so no draft-time optimizer can do much better than a
modest edge — the win comes from *build choice*, which is what this tool makes explicit.

## Example — 2026 optimizer rosters

**balanced** (expected contested 0.877, overall 0.877): Nikola Jokic, Shai Gilgeous-Alexander, Giannis Antetokounmpo, Victor Wembanyama, Luka Doncic, Tyrese Maxey, Anthony Davis, Stephen Curry, Anthony Edwards, James Harden, Karl-Anthony Towns, Derrick White, Brandon Miller
**punt_ft** (expected contested 0.861, overall 0.877): Victor Wembanyama, Nikola Jokic, Shai Gilgeous-Alexander, Luka Doncic, Anthony Davis, Tyrese Maxey, Derrick White, Nikola Vucevic, Stephen Curry, Jalen Johnson, James Harden, Anthony Edwards, Karl-Anthony Towns

Projected per-category win probability (punt_ft):

| cat | PTS | REB | AST | STL | BLK | FG3M | FT | FG | TOV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P(win) | 1.00 | 0.98 | 0.99 | 0.99 | 0.97 | 0.96 | 1.00 | 1.00 | 0.00 |

---
*Caveats:* validation is in-sim (the simulator partly bakes in the archetype→category mapping), so
treat the lifts as *directional* and reality-check against real leagues. The optimizer uses prior-season
z as the projection; a proper next-season projection (Phase 3) is the upgrade that raises the ceiling.
