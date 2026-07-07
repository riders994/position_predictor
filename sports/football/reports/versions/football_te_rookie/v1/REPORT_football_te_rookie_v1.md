# Position Predictor — Results: football TE v1

_**TE v1** · primary @ 2bc2224-dirty_

_Generated 2026-07-06._ Test seasons **[2021, 2022, 2023, 2024, 2025]**, eligibility cutoff **g\* = 1 games**. Ranking is computed within each test season; metrics are mean ± sd across the season folds.

## Headline

- **Best model:** `ridge` (mean, 30-yr window) — Spearman **0.604 ± 0.180**, Precision@12 0.97, MAE 2.82 PPG (full eligible universe).

## Recency: how much history helps (§6.2)

Spearman by training-window length (years), at g\*:

| model | combine | 10yr | 20yr | 30yr |
|---|---|---|---|---|
| elasticnet | mean | 0.422 | 0.502 | 0.574 |
| elasticnet | recency_weighted | 0.422 | 0.502 | 0.528 |
| elasticnet | val_weighted | 0.422 | 0.502 | 0.567 |
| lasso | mean | 0.442 | 0.516 | 0.565 |
| lasso | recency_weighted | 0.442 | 0.516 | 0.525 |
| lasso | val_weighted | 0.442 | 0.516 | 0.562 |
| lightgbm | mean | 0.336 | 0.387 | 0.373 |
| lightgbm | recency_weighted | 0.336 | 0.387 | 0.373 |
| lightgbm | val_weighted | 0.336 | 0.387 | 0.373 |
| random_forest | mean | 0.392 | 0.431 | 0.448 |
| random_forest | recency_weighted | 0.392 | 0.431 | 0.442 |
| random_forest | val_weighted | 0.392 | 0.431 | 0.448 |
| ridge | mean | 0.447 | 0.527 | 0.604 |
| ridge | recency_weighted | 0.447 | 0.527 | 0.553 |
| ridge | val_weighted | 0.447 | 0.527 | 0.577 |
| xgboost | mean | 0.389 | 0.439 | 0.563 |
| xgboost | recency_weighted | 0.389 | 0.439 | 0.525 |
| xgboost | val_weighted | 0.389 | 0.439 | 0.552 |

> 20 yr ≈ 30 yr is expected — the box-score era is data-capped at 1999.

## Market benchmark (§7.4)

_No market benchmark available (run `make benchmark`)._

## NGS-block ablation (§7.3)

_n/a_

## Availability model (§7.4)

Predicting N+1 games played (gates projected eligibility / injury risk):

| model | games MAE | clears-cutoff AUC |
|---|---|---|
| gbm_poisson | 4.03 | 0.840 |

## Eligibility-cutoff sensitivity (§6.4)

Best model (`ridge`) Spearman across the candidate games cutoffs (robustness — the headline does not hinge on g\*):

| cutoff (games) | 1 |
|---|---|
| Spearman | 0.517 |


## Data volume

- **Feature rows:** 389 (369 labeled with a next-season target)
- **Features:** 18 columns
- **Seasons:** 1999–2026 (27 seasons)
- **Labeled rows per era:** capital_only 13, capital_plus_combine 356

## Compute & efficiency

- **Experiment wall-clock:** 9.3s (total model fit time 7.5s across the grid)

Per-model cost vs ranking quality at the headline window (30-yr, g\*=1). **Efficiency** = Spearman per fit-second:

| model | combine | fit (s) | train rows | features | Spearman | P@12 | efficiency |
|---|---|---|---|---|---|---|---|
| ridge | mean | 0.01 | 296 | 18 | 0.604 | 0.97 | 119.31 |
| ridge | val_weighted | 0.02 | 296 | 18 | 0.577 | 0.97 | 37.13 |
| elasticnet | mean | 0.00 | 296 | 18 | 0.574 | 0.95 | 122.86 |
| elasticnet | val_weighted | 0.02 | 296 | 18 | 0.567 | 0.95 | 36.28 |
| lasso | mean | 0.00 | 296 | 18 | 0.565 | 0.95 | 123.57 |
| xgboost | mean | 0.32 | 296 | 18 | 0.563 | 0.95 | 1.78 |
| lasso | val_weighted | 0.02 | 296 | 18 | 0.562 | 0.95 | 37.17 |
| ridge | recency_weighted | 0.01 | 296 | 18 | 0.553 | 0.97 | 109.64 |
| xgboost | val_weighted | 0.65 | 296 | 18 | 0.552 | 0.95 | 0.85 |
| elasticnet | recency_weighted | 0.00 | 296 | 18 | 0.528 | 0.95 | 110.88 |
| xgboost | recency_weighted | 0.32 | 296 | 18 | 0.525 | 0.95 | 1.66 |
| lasso | recency_weighted | 0.00 | 296 | 18 | 0.525 | 0.95 | 111.47 |
| random_forest | val_weighted | 0.53 | 296 | 18 | 0.448 | 0.95 | 0.84 |
| random_forest | mean | 0.22 | 296 | 18 | 0.448 | 0.95 | 2.01 |
| random_forest | recency_weighted | 0.21 | 296 | 18 | 0.442 | 0.95 | 2.07 |
| lightgbm | mean | 0.21 | 296 | 18 | 0.373 | 0.95 | 1.82 |
| lightgbm | val_weighted | 0.33 | 296 | 18 | 0.373 | 0.95 | 1.12 |
| lightgbm | recency_weighted | 0.18 | 296 | 18 | 0.373 | 0.95 | 2.11 |

## Figures

![Spearman by window](figures/report_football_te_rookie_windows.png)

![Model vs market](figures/report_football_te_rookie_vs_market.png)
