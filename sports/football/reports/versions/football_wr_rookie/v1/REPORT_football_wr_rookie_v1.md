# Position Predictor — Results: football WR v1

_**WR v1** · primary @ 2bc2224-dirty_

_Generated 2026-07-06._ Test seasons **[2021, 2022, 2023, 2024, 2025]**, eligibility cutoff **g\* = 1 games**. Ranking is computed within each test season; metrics are mean ± sd across the season folds.

## Headline

- **Best model:** `ridge` (recency_weighted, 30-yr window) — Spearman **0.621 ± 0.044**, Precision@12 0.68, MAE 2.85 PPG (full eligible universe).

## Recency: how much history helps (§6.2)

Spearman by training-window length (years), at g\*:

| model | combine | 10yr | 20yr | 30yr |
|---|---|---|---|---|
| elasticnet | mean | 0.595 | 0.615 | 0.608 |
| elasticnet | recency_weighted | 0.595 | 0.615 | 0.616 |
| elasticnet | val_weighted | 0.595 | 0.615 | 0.611 |
| lasso | mean | 0.593 | 0.618 | 0.609 |
| lasso | recency_weighted | 0.593 | 0.618 | 0.612 |
| lasso | val_weighted | 0.593 | 0.618 | 0.613 |
| lightgbm | mean | 0.457 | 0.473 | 0.463 |
| lightgbm | recency_weighted | 0.457 | 0.473 | 0.463 |
| lightgbm | val_weighted | 0.457 | 0.473 | 0.463 |
| random_forest | mean | 0.527 | 0.538 | 0.609 |
| random_forest | recency_weighted | 0.527 | 0.538 | 0.578 |
| random_forest | val_weighted | 0.527 | 0.538 | 0.586 |
| ridge | mean | 0.592 | 0.621 | 0.612 |
| ridge | recency_weighted | 0.592 | 0.621 | 0.621 |
| ridge | val_weighted | 0.592 | 0.621 | 0.616 |
| xgboost | mean | 0.477 | 0.530 | 0.587 |
| xgboost | recency_weighted | 0.477 | 0.530 | 0.584 |
| xgboost | val_weighted | 0.477 | 0.530 | 0.584 |

> 20 yr ≈ 30 yr is expected — the box-score era is data-capped at 1999.

## Market benchmark (§7.4)

_No market benchmark available (run `make benchmark`)._

## NGS-block ablation (§7.3)

_n/a_

## Availability model (§7.4)

Predicting N+1 games played (gates projected eligibility / injury risk):

| model | games MAE | clears-cutoff AUC |
|---|---|---|
| gbm_poisson | 4.28 | 0.782 |

## Eligibility-cutoff sensitivity (§6.4)

Best model (`ridge`) Spearman across the candidate games cutoffs (robustness — the headline does not hinge on g\*):

| cutoff (games) | 1 |
|---|---|
| Spearman | 0.609 |


## Data volume

- **Feature rows:** 821 (788 labeled with a next-season target)
- **Features:** 18 columns
- **Seasons:** 1999–2026 (27 seasons)
- **Labeled rows per era:** capital_only 25, capital_plus_combine 763

## Compute & efficiency

- **Experiment wall-clock:** 11.7s (total model fit time 9.5s across the grid)

Per-model cost vs ranking quality at the headline window (30-yr, g\*=1). **Efficiency** = Spearman per fit-second:

| model | combine | fit (s) | train rows | features | Spearman | P@12 | efficiency |
|---|---|---|---|---|---|---|---|
| ridge | recency_weighted | 0.01 | 627 | 18 | 0.621 | 0.68 | 113.29 |
| elasticnet | recency_weighted | 0.01 | 627 | 18 | 0.616 | 0.70 | 98.43 |
| ridge | val_weighted | 0.02 | 627 | 18 | 0.616 | 0.70 | 36.88 |
| lasso | val_weighted | 0.02 | 627 | 18 | 0.613 | 0.70 | 32.02 |
| ridge | mean | 0.01 | 627 | 18 | 0.612 | 0.70 | 116.66 |
| lasso | recency_weighted | 0.01 | 627 | 18 | 0.612 | 0.70 | 68.34 |
| elasticnet | val_weighted | 0.02 | 627 | 18 | 0.611 | 0.68 | 35.52 |
| random_forest | mean | 0.25 | 627 | 18 | 0.609 | 0.70 | 2.43 |
| lasso | mean | 0.01 | 627 | 18 | 0.609 | 0.70 | 69.26 |
| elasticnet | mean | 0.01 | 627 | 18 | 0.608 | 0.68 | 107.48 |
| xgboost | mean | 0.33 | 627 | 18 | 0.587 | 0.70 | 1.77 |
| random_forest | val_weighted | 0.58 | 627 | 18 | 0.586 | 0.67 | 1.01 |
| xgboost | val_weighted | 0.66 | 627 | 18 | 0.584 | 0.70 | 0.89 |
| xgboost | recency_weighted | 0.32 | 627 | 18 | 0.584 | 0.70 | 1.84 |
| random_forest | recency_weighted | 0.25 | 627 | 18 | 0.578 | 0.67 | 2.28 |
| lightgbm | mean | 0.37 | 627 | 18 | 0.463 | 0.63 | 1.25 |
| lightgbm | val_weighted | 0.67 | 627 | 18 | 0.463 | 0.63 | 0.69 |
| lightgbm | recency_weighted | 0.36 | 627 | 18 | 0.463 | 0.63 | 1.28 |

## Figures

![Spearman by window](figures/report_football_wr_rookie_windows.png)

![Model vs market](figures/report_football_wr_rookie_vs_market.png)
