# Phase 2 — Archetype Composition → 9-cat Success

**Corpus:** 88 fantasy team-seasons across 8 league-seasons (Yahoo 9-cat redraft,
2015-16…2023-24). **Features:** 12 weeks-weighted archetype soft shares (`comp_*`, sum≈1).
**Target:** `reg_win_pct` (mean 0.500, std 0.195).

## Generalization (leave-one-league-season-out)

| model | out-of-fold R² | out-of-fold MAE | within-league ρ |
| --- | --- | --- | --- |
| baseline (mean) | +0.000 | 0.157 | +nan |
| Ridge | -0.005 | 0.157 | -0.119 |
| Lasso | +0.000 | 0.157 | +nan |
| GBM (depth2) | -0.211 | 0.174 | +0.045 |

> Composition has **no reliable out-of-sample predictive power** at this sample size — the value is directional (signs/contrast), not point prediction. A mean-only baseline has R²=0 by construction; a model adds value only if it beats its MAE
> *or* shows a positive **within-league ρ** (correctly orders a held-out league's teams — the natural
> skill metric for a ranking/zero-sum target).

## Which archetype tilts associate with winning (Ridge, standardized)

Coefficients are *relative tilts away from the average roster mix* (compositional: shares sum to 1),
not independent effects. `sign stability` = fraction of league-resampled bootstraps keeping the sign —
**the trustworthy column at n≈88**.

| archetype | coef (std) | sign stability |
| --- | --- | --- |
| Scoring Combo Guard | +0.0019 | 91% |
| High-Usage Primary | +0.0013 | 84% |
| Lead Playmaker | +0.0008 | 69% |
| Off-Ball Wing | +0.0006 | 66% |
| Interior Big | +0.0003 | 56% |
| Connector Wing | +0.0002 | 55% |
| 3-and-D Wing | -0.0000 | 54% |
| Two-Way Forward | -0.0008 | 80% |
| Slashing Non-Shooter | -0.0010 | 81% |
| Rim-Running Center | -0.0016 | 74% |
| Foul-Drawing Iso Scorer | -0.0020 | 97% |
| Non-Shooting Center | -0.0024 | 92% |

## Top vs bottom success quartile — mean composition

Descriptive and robust: how the most- and least-successful rosters are built, by share.

| archetype | top quartile | bottom quartile | diff |
| --- | --- | --- | --- |
| Scoring Combo Guard | 0.213 | 0.184 | +0.029 |
| Interior Big | 0.079 | 0.069 | +0.010 |
| Off-Ball Wing | 0.083 | 0.076 | +0.007 |
| High-Usage Primary | 0.05 | 0.043 | +0.007 |
| Lead Playmaker | 0.096 | 0.093 | +0.002 |
| Connector Wing | 0.059 | 0.058 | +0.001 |
| Slashing Non-Shooter | 0.059 | 0.06 | -0.001 |
| Non-Shooting Center | 0.0 | 0.004 | -0.003 |
| Two-Way Forward | 0.154 | 0.163 | -0.009 |
| Foul-Drawing Iso Scorer | 0.028 | 0.041 | -0.013 |
| 3-and-D Wing | 0.147 | 0.162 | -0.015 |
| Rim-Running Center | 0.032 | 0.047 | -0.015 |

---
*Caveats:* n≈88 from one manager's league pool (selection); `cat_win_rate` is ~zero-sum within
a league; linear/Ridge won't capture **punt** builds (category concentration) — a concentration
feature + the archetype×category contribution matrix are the next refinements.
