# Position Predictor — Results: football QB v1

_**QB v1** · primary @ 2bc2224-dirty_

_Generated 2026-07-06._ Test seasons **[2021, 2022, 2023, 2024, 2025]**, eligibility cutoff **g\* = 1 games**. Ranking is computed within each test season; metrics are mean ± sd across the season folds.

## Headline

- **Best model:** `random_forest` (recency_weighted, 30-yr window) — Spearman **0.480 ± 0.242**, Precision@12 1.00, MAE 6.77 PPG (full eligible universe).

## Recency: how much history helps (§6.2)

Spearman by training-window length (years), at g\*:

| model | combine | 10yr | 20yr | 30yr |
|---|---|---|---|---|
| elasticnet | mean | 0.412 | 0.412 | 0.378 |
| elasticnet | recency_weighted | 0.412 | 0.412 | 0.413 |
| elasticnet | val_weighted | 0.412 | 0.412 | 0.413 |
| lasso | mean | 0.412 | 0.412 | 0.378 |
| lasso | recency_weighted | 0.412 | 0.412 | 0.413 |
| lasso | val_weighted | 0.412 | 0.412 | 0.413 |
| lightgbm | mean | 0.330 | 0.342 | 0.382 |
| lightgbm | recency_weighted | 0.330 | 0.342 | 0.382 |
| lightgbm | val_weighted | 0.330 | 0.342 | 0.382 |
| random_forest | mean | 0.472 | 0.462 | 0.431 |
| random_forest | recency_weighted | 0.472 | 0.462 | 0.480 |
| random_forest | val_weighted | 0.472 | 0.462 | 0.431 |
| ridge | mean | 0.399 | 0.399 | 0.413 |
| ridge | recency_weighted | 0.399 | 0.399 | 0.420 |
| ridge | val_weighted | 0.399 | 0.399 | 0.413 |
| xgboost | mean | 0.386 | 0.312 | 0.332 |
| xgboost | recency_weighted | 0.386 | 0.312 | 0.301 |
| xgboost | val_weighted | 0.386 | 0.312 | 0.332 |

> 20 yr ≈ 30 yr is expected — the box-score era is data-capped at 1999.

## Market benchmark (§7.4)

_No market benchmark available (run `make benchmark`)._

## NGS-block ablation (§7.3)

_n/a_

## Availability model (§7.4)

Predicting N+1 games played (gates projected eligibility / injury risk):

| model | games MAE | clears-cutoff AUC |
|---|---|---|
| gbm_poisson | 3.74 | 0.693 |

## Eligibility-cutoff sensitivity (§6.4)

Best model (`random_forest`) Spearman across the candidate games cutoffs (robustness — the headline does not hinge on g\*):

| cutoff (games) | 1 |
|---|---|
| Spearman | 0.461 |


## Data volume

- **Feature rows:** 306 (297 labeled with a next-season target)
- **Features:** 18 columns
- **Seasons:** 1999–2026 (27 seasons)
- **Labeled rows per era:** capital_only 12, capital_plus_combine 285

## Compute & efficiency

- **Experiment wall-clock:** 8.6s (total model fit time 6.9s across the grid)

Per-model cost vs ranking quality at the headline window (30-yr, g\*=1). **Efficiency** = Spearman per fit-second:

| model | combine | fit (s) | train rows | features | Spearman | P@12 | efficiency |
|---|---|---|---|---|---|---|---|
| random_forest | recency_weighted | 0.22 | 239 | 18 | 0.480 | 1.00 | 2.23 |
| random_forest | mean | 0.22 | 239 | 18 | 0.431 | 1.00 | 1.99 |
| random_forest | val_weighted | 0.53 | 239 | 18 | 0.431 | 1.00 | 0.81 |
| ridge | recency_weighted | 0.00 | 239 | 18 | 0.420 | 1.00 | 87.84 |
| lasso | val_weighted | 0.02 | 239 | 18 | 0.413 | 1.00 | 26.44 |
| elasticnet | recency_weighted | 0.00 | 239 | 18 | 0.413 | 1.00 | 94.16 |
| lasso | recency_weighted | 0.00 | 239 | 18 | 0.413 | 1.00 | 94.01 |
| ridge | val_weighted | 0.02 | 239 | 18 | 0.413 | 1.00 | 23.34 |
| ridge | mean | 0.01 | 239 | 18 | 0.413 | 1.00 | 82.03 |
| elasticnet | val_weighted | 0.01 | 239 | 18 | 0.413 | 1.00 | 27.69 |
| lightgbm | mean | 0.15 | 239 | 18 | 0.382 | 1.00 | 2.51 |
| lightgbm | val_weighted | 0.29 | 239 | 18 | 0.382 | 1.00 | 1.33 |
| lightgbm | recency_weighted | 0.16 | 239 | 18 | 0.382 | 1.00 | 2.46 |
| elasticnet | mean | 0.00 | 239 | 18 | 0.378 | 1.00 | 80.60 |
| lasso | mean | 0.00 | 239 | 18 | 0.378 | 1.00 | 78.77 |
| xgboost | mean | 0.31 | 239 | 18 | 0.332 | 1.00 | 1.08 |
| xgboost | val_weighted | 0.63 | 239 | 18 | 0.332 | 1.00 | 0.53 |
| xgboost | recency_weighted | 0.31 | 239 | 18 | 0.301 | 1.00 | 0.98 |

## Figures

![Spearman by window](figures/report_football_qb_rookie_windows.png)

![Model vs market](figures/report_football_qb_rookie_vs_market.png)
