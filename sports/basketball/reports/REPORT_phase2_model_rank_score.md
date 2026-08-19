# Phase 2 — Archetype Composition → 9-cat Success

**Corpus:** 88 fantasy team-seasons across 8 league-seasons (Yahoo 9-cat redraft,
2015-16…2023-24). **Features:** 12 weeks-weighted archetype soft shares (`comp_*`, sum≈1).
**Target:** `rank_score` (mean 0.500, std 0.318).

## Generalization (leave-one-league-season-out)

| model | out-of-fold R² | out-of-fold MAE | within-league ρ |
| --- | --- | --- | --- |
| baseline (mean) | +0.000 | 0.275 | +nan |
| Ridge | -0.030 | 0.277 | +0.094 |
| Lasso | -0.042 | 0.280 | -0.399 |
| GBM (depth2) | -0.191 | 0.291 | +0.106 |

> Composition has **no reliable out-of-sample predictive power** at this sample size — the value is directional (signs/contrast), not point prediction. A mean-only baseline has R²=0 by construction; a model adds value only if it beats its MAE
> *or* shows a positive **within-league ρ** (correctly orders a held-out league's teams — the natural
> skill metric for a ranking/zero-sum target).

## Which archetype tilts associate with winning (Ridge, standardized)

Coefficients are *relative tilts away from the average roster mix* (compositional: shares sum to 1),
not independent effects. `sign stability` = fraction of league-resampled bootstraps keeping the sign —
**the trustworthy column at n≈88**.

| archetype | coef (std) | sign stability |
| --- | --- | --- |
| Stretch Forward | +0.0092 | 89% |
| Lead Playmaker | +0.0081 | 98% |
| Low-Usage Wing | +0.0056 | 90% |
| High-Usage Engine | +0.0024 | 74% |
| Lead Scoring Guard | +0.0023 | 70% |
| Wing Shot-Creator | +0.0003 | 56% |
| Rebounding Forward | -0.0007 | 63% |
| Rim-Running Center | -0.0012 | 55% |
| Movement Shooter | -0.0021 | 68% |
| Perimeter Stopper | -0.0040 | 77% |
| Non-Scoring Playmaker | -0.0043 | 72% |
| Rim-Protecting Big | -0.0065 | 95% |
| Foul-Drawing Iso Scorer | -0.0079 | 96% |

## Top vs bottom success quartile — mean composition

Descriptive and robust: how the most- and least-successful rosters are built, by share.

| archetype | top quartile | bottom quartile | diff |
| --- | --- | --- | --- |
| Lead Playmaker | 0.109 | 0.088 | +0.021 |
| Stretch Forward | 0.094 | 0.075 | +0.019 |
| Lead Scoring Guard | 0.146 | 0.129 | +0.016 |
| Low-Usage Wing | 0.033 | 0.027 | +0.007 |
| High-Usage Engine | 0.061 | 0.059 | +0.002 |
| Wing Shot-Creator | 0.12 | 0.119 | +0.001 |
| Rebounding Forward | 0.071 | 0.073 | -0.002 |
| Rim-Running Center | 0.091 | 0.095 | -0.004 |
| Non-Scoring Playmaker | 0.027 | 0.036 | -0.009 |
| Perimeter Stopper | 0.056 | 0.066 | -0.010 |
| Movement Shooter | 0.085 | 0.095 | -0.010 |
| Foul-Drawing Iso Scorer | 0.008 | 0.022 | -0.014 |
| Rim-Protecting Big | 0.098 | 0.115 | -0.017 |

---
*Caveats:* n≈88 from one manager's league pool (selection); `cat_win_rate` is ~zero-sum within
a league; linear/Ridge won't capture **punt** builds (category concentration) — a concentration
feature + the archetype×category contribution matrix are the next refinements.
