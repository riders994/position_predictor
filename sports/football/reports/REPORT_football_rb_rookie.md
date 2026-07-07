# Position Predictor — Results: football RB v1

_**RB v1** · primary @ 2bc2224-dirty_

_Generated 2026-07-06._ Test seasons **[2021, 2022, 2023, 2024, 2025]**, eligibility cutoff **g\* = 1 games**. Ranking is computed within each test season; metrics are mean ± sd across the season folds.

## Headline

- **Best model:** `lasso` (mean, 30-yr window) — Spearman **0.536 ± 0.186**, Precision@12 0.73, MAE 3.29 PPG (full eligible universe).

## Recency: how much history helps (§6.2)

Spearman by training-window length (years), at g\*:

| model | combine | 10yr | 20yr | 30yr |
|---|---|---|---|---|
| elasticnet | mean | 0.515 | 0.509 | 0.531 |
| elasticnet | recency_weighted | 0.515 | 0.509 | 0.520 |
| elasticnet | val_weighted | 0.515 | 0.509 | 0.522 |
| lasso | mean | 0.515 | 0.518 | 0.536 |
| lasso | recency_weighted | 0.515 | 0.518 | 0.512 |
| lasso | val_weighted | 0.515 | 0.518 | 0.528 |
| lightgbm | mean | 0.443 | 0.404 | 0.409 |
| lightgbm | recency_weighted | 0.443 | 0.404 | 0.409 |
| lightgbm | val_weighted | 0.443 | 0.404 | 0.409 |
| random_forest | mean | 0.428 | 0.410 | 0.451 |
| random_forest | recency_weighted | 0.428 | 0.410 | 0.443 |
| random_forest | val_weighted | 0.428 | 0.410 | 0.452 |
| ridge | mean | 0.530 | 0.509 | 0.527 |
| ridge | recency_weighted | 0.530 | 0.509 | 0.521 |
| ridge | val_weighted | 0.530 | 0.509 | 0.519 |
| xgboost | mean | 0.425 | 0.394 | 0.462 |
| xgboost | recency_weighted | 0.425 | 0.394 | 0.500 |
| xgboost | val_weighted | 0.425 | 0.394 | 0.497 |

> 20 yr ≈ 30 yr is expected — the box-score era is data-capped at 1999.

## Market benchmark (§7.4)

_No market benchmark available (run `make benchmark`)._

## NGS-block ablation (§7.3)

_n/a_

## Availability model (§7.4)

Predicting N+1 games played (gates projected eligibility / injury risk):

| model | games MAE | clears-cutoff AUC |
|---|---|---|
| gbm_poisson | 5.04 | 0.643 |

## Eligibility-cutoff sensitivity (§6.4)

Best model (`lasso`) Spearman across the candidate games cutoffs (robustness — the headline does not hinge on g\*):

| cutoff (games) | 1 |
|---|---|
| Spearman | 0.520 |


## Data volume

- **Feature rows:** 585 (573 labeled with a next-season target)
- **Features:** 18 columns
- **Seasons:** 1999–2026 (27 seasons)
- **Labeled rows per era:** capital_only 25, capital_plus_combine 548

## Compute & efficiency

- **Experiment wall-clock:** 10.7s (total model fit time 8.6s across the grid)

Per-model cost vs ranking quality at the headline window (30-yr, g\*=1). **Efficiency** = Spearman per fit-second:

| model | combine | fit (s) | train rows | features | Spearman | P@12 | efficiency |
|---|---|---|---|---|---|---|---|
| lasso | mean | 0.00 | 468 | 18 | 0.536 | 0.73 | 115.21 |
| elasticnet | mean | 0.00 | 468 | 18 | 0.531 | 0.73 | 114.77 |
| lasso | val_weighted | 0.02 | 468 | 18 | 0.528 | 0.73 | 33.74 |
| ridge | mean | 0.00 | 468 | 18 | 0.527 | 0.73 | 105.80 |
| elasticnet | val_weighted | 0.02 | 468 | 18 | 0.522 | 0.73 | 33.62 |
| ridge | recency_weighted | 0.01 | 468 | 18 | 0.521 | 0.77 | 103.91 |
| elasticnet | recency_weighted | 0.00 | 468 | 18 | 0.520 | 0.73 | 112.42 |
| ridge | val_weighted | 0.02 | 468 | 18 | 0.519 | 0.73 | 32.47 |
| lasso | recency_weighted | 0.00 | 468 | 18 | 0.512 | 0.73 | 106.18 |
| xgboost | recency_weighted | 0.32 | 468 | 18 | 0.500 | 0.73 | 1.57 |
| xgboost | val_weighted | 0.66 | 468 | 18 | 0.497 | 0.72 | 0.76 |
| xgboost | mean | 0.32 | 468 | 18 | 0.462 | 0.72 | 1.44 |
| random_forest | val_weighted | 0.55 | 468 | 18 | 0.452 | 0.73 | 0.82 |
| random_forest | mean | 0.23 | 468 | 18 | 0.451 | 0.73 | 1.96 |
| random_forest | recency_weighted | 0.23 | 468 | 18 | 0.443 | 0.73 | 1.89 |
| lightgbm | mean | 0.27 | 468 | 18 | 0.409 | 0.72 | 1.51 |
| lightgbm | val_weighted | 0.51 | 468 | 18 | 0.409 | 0.72 | 0.81 |
| lightgbm | recency_weighted | 0.28 | 468 | 18 | 0.409 | 0.72 | 1.47 |

## Figures

![Spearman by window](figures/report_football_rb_rookie_windows.png)

![Model vs market](figures/report_football_rb_rookie_vs_market.png)
