# Phase 2 — Archetype Composition → 9-cat Success

**Corpus:** 88 fantasy team-seasons across 8 league-seasons (Yahoo 9-cat redraft,
2015-16…2023-24). **Features:** 12 weeks-weighted archetype soft shares (`comp_*`, sum≈1).
**Target:** `rank_score` (mean 0.500, std 0.318).

## Generalization (leave-one-league-season-out)

| model | out-of-fold R² | out-of-fold MAE | within-league ρ |
| --- | --- | --- | --- |
| baseline (mean) | +0.000 | 0.275 | +nan |
| Ridge | -0.006 | 0.275 | -0.138 |
| Lasso | -0.009 | 0.275 | -0.148 |
| GBM (depth2) | -0.192 | 0.281 | +0.059 |

> Composition has **no reliable out-of-sample predictive power** at this sample size — the value is directional (signs/contrast), not point prediction. A mean-only baseline has R²=0 by construction; a model adds value only if it beats its MAE
> *or* shows a positive **within-league ρ** (correctly orders a held-out league's teams — the natural
> skill metric for a ranking/zero-sum target).

## Which archetype tilts associate with winning (Ridge, standardized)

Coefficients are *relative tilts away from the average roster mix* (compositional: shares sum to 1),
not independent effects. `sign stability` = fraction of league-resampled bootstraps keeping the sign —
**the trustworthy column at n≈88**.

| archetype | coef (std) | sign stability |
| --- | --- | --- |
| Scoring Combo Guard | +0.0024 | 84% |
| Lead Playmaker | +0.0009 | 61% |
| High-Usage Primary | +0.0007 | 66% |
| Off-Ball Wing | +0.0005 | 58% |
| Interior Big | +0.0004 | 60% |
| Slashing Non-Shooter | +0.0000 | 52% |
| Two-Way Forward | -0.0002 | 53% |
| Rim-Running Center | -0.0004 | 51% |
| 3-and-D Wing | -0.0008 | 62% |
| Connector Wing | -0.0010 | 65% |
| Foul-Drawing Iso Scorer | -0.0032 | 94% |
| Non-Shooting Center | -0.0045 | 99% |

## Top vs bottom success quartile — mean composition

Descriptive and robust: how the most- and least-successful rosters are built, by share.

| archetype | top quartile | bottom quartile | diff |
| --- | --- | --- | --- |
| Scoring Combo Guard | 0.22 | 0.19 | +0.030 |
| Interior Big | 0.08 | 0.069 | +0.012 |
| Lead Playmaker | 0.104 | 0.102 | +0.002 |
| Two-Way Forward | 0.156 | 0.155 | +0.001 |
| Off-Ball Wing | 0.075 | 0.075 | +0.000 |
| High-Usage Primary | 0.047 | 0.05 | -0.002 |
| Rim-Running Center | 0.043 | 0.047 | -0.003 |
| Slashing Non-Shooter | 0.055 | 0.058 | -0.003 |
| Connector Wing | 0.054 | 0.059 | -0.005 |
| Non-Shooting Center | 0.001 | 0.005 | -0.005 |
| Foul-Drawing Iso Scorer | 0.028 | 0.04 | -0.013 |
| 3-and-D Wing | 0.137 | 0.151 | -0.014 |

---
*Caveats:* n≈88 from one manager's league pool (selection); `cat_win_rate` is ~zero-sum within
a league; linear/Ridge won't capture **punt** builds (category concentration) — a concentration
feature + the archetype×category contribution matrix are the next refinements.
