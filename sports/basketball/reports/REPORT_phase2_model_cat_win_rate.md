# Phase 2 — Archetype Composition → 9-cat Success

**Corpus:** 88 fantasy team-seasons across 8 league-seasons (Yahoo 9-cat redraft,
2015-16…2023-24). **Features:** 12 weeks-weighted archetype soft shares (`comp_*`, sum≈1).
**Target:** `cat_win_rate` (mean 0.500, std 0.083).

## Generalization (leave-one-league-season-out)

| model | out-of-fold R² | out-of-fold MAE | within-league ρ |
| --- | --- | --- | --- |
| baseline (mean) | -0.000 | 0.064 | +nan |
| Ridge | -0.079 | 0.065 | -0.077 |
| Lasso | -0.000 | 0.064 | +nan |
| GBM (depth2) | -0.170 | 0.070 | +0.033 |

> Composition has **no reliable out-of-sample predictive power** at this sample size — the value is directional (signs/contrast), not point prediction. A mean-only baseline has R²=0 by construction; a model adds value only if it beats its MAE
> *or* shows a positive **within-league ρ** (correctly orders a held-out league's teams — the natural
> skill metric for a ranking/zero-sum target).

## Which archetype tilts associate with winning (Ridge, standardized)

Coefficients are *relative tilts away from the average roster mix* (compositional: shares sum to 1),
not independent effects. `sign stability` = fraction of league-resampled bootstraps keeping the sign —
**the trustworthy column at n≈88**.

| archetype | coef (std) | sign stability |
| --- | --- | --- |
| Lead Playmaker | +0.0012 | 99% |
| Stretch Forward | +0.0009 | 81% |
| Low-Usage Wing | +0.0009 | 96% |
| High-Usage Engine | +0.0005 | 85% |
| Wing Shot-Creator | +0.0004 | 77% |
| Perimeter Stopper | +0.0002 | 70% |
| Movement Shooter | +0.0001 | 60% |
| Rim-Running Center | +0.0001 | 62% |
| Lead Scoring Guard | -0.0005 | 71% |
| Non-Scoring Playmaker | -0.0007 | 65% |
| Foul-Drawing Iso Scorer | -0.0008 | 88% |
| Rebounding Forward | -0.0009 | 87% |
| Rim-Protecting Big | -0.0009 | 81% |

## Top vs bottom success quartile — mean composition

Descriptive and robust: how the most- and least-successful rosters are built, by share.

| archetype | top quartile | bottom quartile | diff |
| --- | --- | --- | --- |
| Lead Playmaker | 0.107 | 0.087 | +0.020 |
| Stretch Forward | 0.091 | 0.08 | +0.011 |
| Wing Shot-Creator | 0.122 | 0.112 | +0.009 |
| Rim-Running Center | 0.088 | 0.083 | +0.006 |
| High-Usage Engine | 0.062 | 0.056 | +0.006 |
| Movement Shooter | 0.093 | 0.086 | +0.006 |
| Low-Usage Wing | 0.035 | 0.031 | +0.004 |
| Lead Scoring Guard | 0.138 | 0.14 | -0.002 |
| Perimeter Stopper | 0.06 | 0.064 | -0.004 |
| Non-Scoring Playmaker | 0.027 | 0.034 | -0.008 |
| Foul-Drawing Iso Scorer | 0.008 | 0.017 | -0.009 |
| Rim-Protecting Big | 0.107 | 0.126 | -0.019 |
| Rebounding Forward | 0.063 | 0.083 | -0.020 |

---
*Caveats:* n≈88 from one manager's league pool (selection); `cat_win_rate` is ~zero-sum within
a league; linear/Ridge won't capture **punt** builds (category concentration) — a concentration
feature + the archetype×category contribution matrix are the next refinements.
