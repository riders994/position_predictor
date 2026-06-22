# Position Predictor — Results: football QB v1

_**QB v1** · qb-model @ 5cba355-dirty_

_Generated 2026-06-17._ Test seasons **[2020, 2021, 2022, 2023, 2024]**, eligibility cutoff **g\* = 7 games**. Ranking is computed within each test season; metrics are mean ± sd across the season folds.

## Headline

- **Best model:** `random_forest` (mean, 20-yr window) — Spearman **0.697 ± 0.045**, Precision@12 0.63, MAE 4.18 PPG (full eligible universe).
- **Best baseline:** `smoothed_history` — Spearman 0.635 (the must-beat floor).
- **Market head-to-head** (FantasyPros preseason ECR, scored on the identical rows the market ranks): market Spearman 0.695 vs our best `lasso` 0.629 — the model does **not** beat the market on overall rank.
  On **Precision@6 (tier-1 / QB1)**: market 0.53 vs best `xgboost` 0.53 — model trails market.
  On **Precision@12 (tier-2 / QB2)**: market 0.65 vs best `random_forest` 0.67 — model beats market.
  On the **top-weighted** rank score (Weighted τ — errors near #1 count most): market 0.596 vs `random_forest` 0.587 — the model does **not** beat the market where it matters most.

## Recency: how much history helps (§6.2)

Spearman by training-window length (years), at g\*:

| model | combine | 10yr | 20yr | 30yr |
|---|---|---|---|---|
| elasticnet | mean | 0.695 | 0.695 | 0.695 |
| elasticnet | recency_weighted | 0.689 | 0.693 | 0.693 |
| elasticnet | val_weighted | 0.695 | 0.695 | 0.695 |
| lasso | mean | 0.694 | 0.697 | 0.697 |
| lasso | recency_weighted | 0.691 | 0.695 | 0.695 |
| lasso | val_weighted | 0.694 | 0.697 | 0.697 |
| lightgbm | mean | 0.678 | 0.690 | 0.690 |
| lightgbm | recency_weighted | 0.666 | 0.676 | 0.676 |
| lightgbm | val_weighted | 0.680 | 0.691 | 0.691 |
| random_forest | mean | 0.695 | 0.697 | 0.697 |
| random_forest | recency_weighted | 0.690 | 0.693 | 0.693 |
| random_forest | val_weighted | 0.695 | 0.697 | 0.697 |
| ridge | mean | 0.677 | 0.686 | 0.686 |
| ridge | recency_weighted | 0.671 | 0.674 | 0.674 |
| ridge | val_weighted | 0.678 | 0.687 | 0.687 |
| xgboost | mean | 0.693 | 0.693 | 0.693 |
| xgboost | recency_weighted | 0.677 | 0.680 | 0.680 |
| xgboost | val_weighted | 0.694 | 0.693 | 0.693 |

> 20 yr ≈ 30 yr is expected — the box-score era is data-capped at 1999.

## Market benchmark (§7.4)

Preseason ECR coverage of the eligible universe and the market's own ranking quality, per test season:

| season | eligible | ranked | coverage | market Spearman | market P@6 |
|---|---|---|---|---|---|
| 2020 | 57 | 40 | 0.70 | 0.637 | 0.33 |
| 2021 | 61 | 36 | 0.59 | 0.833 | 0.67 |
| 2022 | 67 | 40 | 0.60 | 0.684 | 0.67 |
| 2023 | 63 | 41 | 0.65 | 0.615 | 0.50 |
| 2024 | 63 | 55 | 0.87 | 0.708 | 0.50 |

**Head-to-head on the identical ranked rows** (mean across folds). `weighted_tau` is the **top-weighted** rank score — errors near #1 count most:

| model | Spearman | Weighted τ (top) | Precision@6 (QB1) | Precision@12 (QB2) |
|---|---|---|---|---|
| market_ecr _(market)_ | 0.695 | 0.596 | 0.53 | 0.65 |
| random_forest | 0.618 | 0.587 | 0.50 | 0.67 |
| xgboost | 0.616 | 0.552 | 0.53 | 0.62 |
| lightgbm | 0.610 | 0.551 | 0.47 | 0.63 |
| lasso | 0.629 | 0.518 | 0.50 | 0.62 |
| elasticnet | 0.620 | 0.505 | 0.50 | 0.62 |
| ridge | 0.619 | 0.464 | 0.47 | 0.58 |

## NGS-block ablation (§7.3)

`ngs` era model, longest window — **with** NGS: Spearman 0.618, P@12 0.53; **without**: Spearman 0.618, P@12 0.53.
- **Decision:** drop the `ngs_efficiency` block — no top-12 gain out-of-fold, so the ngs era collapses to the snaps schema (coverage flags retained either way).

## Availability model (§7.4)

Predicting N+1 games played (gates projected eligibility / injury risk):

| model | games MAE | clears-cutoff AUC |
|---|---|---|
| baseline_prior_games | 3.67 | 0.842 |
| gbm_poisson | 3.58 | 0.881 |

## Eligibility-cutoff sensitivity (§6.4)

Best model (`random_forest`) Spearman across the candidate games cutoffs (robustness — the headline does not hinge on g\*):

| cutoff (games) | 6 | 7 | 8 | 10 | 12 | 14 |
|---|---|---|---|---|---|---|
| Spearman | 0.672 | 0.695 | 0.606 | 0.575 | 0.539 | 0.508 |


## Data volume

- **Feature rows:** 1999 (1456 labeled with a next-season target)
- **Features:** 73 columns
- **Seasons:** 1999–2024 (26 seasons)
- **Labeled rows per era:** boxscore 763, snaps 224, ngs 469

## Compute & efficiency

- **Experiment wall-clock:** 64.2s (total model fit time 37.0s across the grid)

Per-model cost vs ranking quality at the headline window (30-yr, g\*=7). **Efficiency** = Spearman per fit-second:

| model | combine | fit (s) | train rows | features | Spearman | P@12 | efficiency |
|---|---|---|---|---|---|---|---|
| random_forest | mean | 1.38 | 1145 | 73 | 0.697 | 0.63 | 0.51 |
| lasso | val_weighted | 0.12 | 1145 | 73 | 0.697 | 0.58 | 5.98 |
| random_forest | val_weighted | 2.80 | 1145 | 73 | 0.697 | 0.63 | 0.25 |
| lasso | mean | 0.05 | 1145 | 73 | 0.697 | 0.58 | 13.16 |
| lasso | recency_weighted | 0.06 | 1145 | 73 | 0.695 | 0.58 | 12.05 |
| elasticnet | val_weighted | 0.12 | 1145 | 73 | 0.695 | 0.58 | 5.93 |
| elasticnet | mean | 0.05 | 1145 | 73 | 0.695 | 0.58 | 12.76 |
| xgboost | val_weighted | 2.18 | 1145 | 73 | 0.693 | 0.58 | 0.32 |
| elasticnet | recency_weighted | 0.05 | 1145 | 73 | 0.693 | 0.58 | 12.74 |
| xgboost | mean | 1.21 | 1145 | 73 | 0.693 | 0.58 | 0.57 |
| random_forest | recency_weighted | 1.41 | 1145 | 73 | 0.693 | 0.60 | 0.49 |
| lightgbm | val_weighted | 0.95 | 1145 | 73 | 0.691 | 0.60 | 0.73 |
| lightgbm | mean | 0.57 | 1145 | 73 | 0.690 | 0.60 | 1.21 |
| ridge | val_weighted | 0.10 | 1145 | 73 | 0.687 | 0.55 | 6.74 |
| ridge | mean | 0.05 | 1145 | 73 | 0.686 | 0.55 | 14.77 |
| xgboost | recency_weighted | 1.45 | 1145 | 73 | 0.680 | 0.60 | 0.47 |
| lightgbm | recency_weighted | 0.59 | 1145 | 73 | 0.676 | 0.58 | 1.14 |
| ridge | recency_weighted | 0.05 | 1145 | 73 | 0.674 | 0.53 | 13.73 |

## Figures

![Spearman by window](figures/report_football_qb_windows.png)

![Model vs market](figures/report_football_qb_vs_market.png)
