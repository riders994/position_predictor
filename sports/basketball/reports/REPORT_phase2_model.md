# Phase 2 — Archetype Composition → 9-cat Success

**Corpus:** 88 fantasy team-seasons across 8 league-seasons (Yahoo 9-cat redraft,
2015-16…2023-24). **Features:** 12 weeks-weighted archetype soft shares (`comp_*`, sum≈1).
**Target:** `cat_win_rate` (mean 0.500, std 0.083).

## Generalization (leave-one-league-season-out)

| model | out-of-fold R² | out-of-fold MAE |
| --- | --- | --- |
| baseline (mean) | -0.000 | 0.064 |
| Ridge | -0.011 | 0.064 |
| Lasso | -0.000 | 0.064 |
| GBM (depth2) | -0.372 | 0.073 |

> Composition has **no reliable out-of-sample predictive power** at this sample size — the value is directional (signs/contrast), not point prediction. A mean-only baseline has R²=0 by construction; a model must beat its MAE to add value.

## Which archetype tilts associate with winning (Ridge, standardized)

Coefficients are *relative tilts away from the average roster mix* (compositional: shares sum to 1),
not independent effects. `sign stability` = fraction of league-resampled bootstraps keeping the sign —
**the trustworthy column at n≈88**.

| archetype | coef (std) | sign stability |
| --- | --- | --- |
| Connector Wing | +0.0004 | 75% |
| Scoring Combo Guard | +0.0004 | 81% |
| Lead Playmaker | +0.0003 | 71% |
| 3-and-D Wing | +0.0003 | 63% |
| Off-Ball Wing | +0.0000 | 56% |
| High-Usage Primary | +0.0000 | 52% |
| Rim-Running Center | -0.0001 | 52% |
| Slashing Non-Shooter | -0.0002 | 69% |
| Interior Big | -0.0002 | 67% |
| Two-Way Forward | -0.0003 | 69% |
| Non-Shooting Center | -0.0007 | 71% |
| Foul-Drawing Iso Scorer | -0.0008 | 95% |

## Top vs bottom success quartile — mean composition

Descriptive and robust: how the most- and least-successful rosters are built, by share.

| archetype | top quartile | bottom quartile | diff |
| --- | --- | --- | --- |
| Scoring Combo Guard | 0.217 | 0.198 | +0.019 |
| Lead Playmaker | 0.105 | 0.092 | +0.014 |
| 3-and-D Wing | 0.145 | 0.139 | +0.005 |
| Slashing Non-Shooter | 0.055 | 0.05 | +0.004 |
| Rim-Running Center | 0.039 | 0.04 | -0.001 |
| Non-Shooting Center | 0.0 | 0.003 | -0.003 |
| Off-Ball Wing | 0.079 | 0.083 | -0.004 |
| Connector Wing | 0.056 | 0.06 | -0.004 |
| High-Usage Primary | 0.045 | 0.05 | -0.005 |
| Interior Big | 0.073 | 0.079 | -0.006 |
| Foul-Drawing Iso Scorer | 0.026 | 0.035 | -0.009 |
| Two-Way Forward | 0.161 | 0.171 | -0.010 |

---
*Caveats:* n≈88 from one manager's league pool (selection); `cat_win_rate` is ~zero-sum within
a league; linear/Ridge won't capture **punt** builds (category concentration) — a concentration
feature + the archetype×category contribution matrix are the next refinements.
