# Phase 2 — Archetype Composition → 9-cat Success

**Corpus:** 88 fantasy team-seasons across 8 league-seasons (Yahoo 9-cat redraft,
2015-16…2023-24). **Features:** 12 weeks-weighted archetype soft shares (`comp_*`, sum≈1).
**Target:** `reg_win_pct` (mean 0.500, std 0.195).

## Generalization (leave-one-league-season-out)

| model | out-of-fold R² | out-of-fold MAE | within-league ρ |
| --- | --- | --- | --- |
| baseline (mean) | +0.000 | 0.157 | +nan |
| Ridge | -0.062 | 0.160 | +0.009 |
| Lasso | -0.024 | 0.158 | -0.600 |
| GBM (depth2) | -0.246 | 0.177 | +0.000 |

> Composition has **no reliable out-of-sample predictive power** at this sample size — the value is directional (signs/contrast), not point prediction. A mean-only baseline has R²=0 by construction; a model adds value only if it beats its MAE
> *or* shows a positive **within-league ρ** (correctly orders a held-out league's teams — the natural
> skill metric for a ranking/zero-sum target).

## Which archetype tilts associate with winning (Ridge, standardized)

Coefficients are *relative tilts away from the average roster mix* (compositional: shares sum to 1),
not independent effects. `sign stability` = fraction of league-resampled bootstraps keeping the sign —
**the trustworthy column at n≈88**.

| archetype | coef (std) | sign stability |
| --- | --- | --- |
| Lead Playmaker | +0.0035 | 99% |
| Stretch Forward | +0.0024 | 85% |
| Low-Usage Wing | +0.0022 | 97% |
| High-Usage Engine | +0.0016 | 89% |
| Wing Shot-Creator | +0.0010 | 72% |
| Lead Scoring Guard | +0.0004 | 57% |
| Movement Shooter | -0.0003 | 52% |
| Rebounding Forward | -0.0011 | 75% |
| Perimeter Stopper | -0.0011 | 70% |
| Rim-Running Center | -0.0013 | 62% |
| Foul-Drawing Iso Scorer | -0.0016 | 79% |
| Non-Scoring Playmaker | -0.0021 | 69% |
| Rim-Protecting Big | -0.0025 | 85% |

## Top vs bottom success quartile — mean composition

Descriptive and robust: how the most- and least-successful rosters are built, by share.

| archetype | top quartile | bottom quartile | diff |
| --- | --- | --- | --- |
| Lead Playmaker | 0.102 | 0.078 | +0.024 |
| High-Usage Engine | 0.064 | 0.051 | +0.014 |
| Lead Scoring Guard | 0.131 | 0.12 | +0.011 |
| Wing Shot-Creator | 0.124 | 0.117 | +0.007 |
| Low-Usage Wing | 0.033 | 0.027 | +0.006 |
| Stretch Forward | 0.089 | 0.084 | +0.005 |
| Rebounding Forward | 0.072 | 0.077 | -0.005 |
| Rim-Running Center | 0.083 | 0.09 | -0.007 |
| Perimeter Stopper | 0.06 | 0.067 | -0.007 |
| Movement Shooter | 0.097 | 0.105 | -0.008 |
| Non-Scoring Playmaker | 0.028 | 0.037 | -0.009 |
| Foul-Drawing Iso Scorer | 0.01 | 0.024 | -0.014 |
| Rim-Protecting Big | 0.107 | 0.123 | -0.016 |

---
*Caveats:* n≈88 from one manager's league pool (selection); `cat_win_rate` is ~zero-sum within
a league; linear/Ridge won't capture **punt** builds (category concentration) — a concentration
feature + the archetype×category contribution matrix are the next refinements.
