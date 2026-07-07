# Position Predictor — Results: football DST v1

_**DST v1** · primary @ 2bc2224-dirty_

_Generated 2026-07-06._ Test seasons **[2021, 2022, 2023, 2024, 2025]**, eligibility cutoff **g\* = 1 games**. Ranking is computed within each test season; metrics are mean ± sd across the season folds.

## Headline

- **Best model:** `lasso` (mean, 20-yr window) — Spearman **0.328 ± 0.196**, Precision@12 0.52, MAE 1.72 PPG (full eligible universe).
- **Best baseline:** `smoothed_history` — Spearman 0.310 (the must-beat floor).

## Recency: how much history helps (§6.2)

Spearman by training-window length (years), at g\*:

| model | combine | 10yr | 20yr | 30yr |
|---|---|---|---|---|
| elasticnet | mean | 0.277 | 0.316 | 0.302 |
| elasticnet | recency_weighted | 0.277 | 0.316 | 0.302 |
| elasticnet | val_weighted | 0.277 | 0.316 | 0.302 |
| lasso | mean | 0.290 | 0.328 | 0.315 |
| lasso | recency_weighted | 0.290 | 0.328 | 0.315 |
| lasso | val_weighted | 0.290 | 0.328 | 0.315 |
| lightgbm | mean | 0.240 | 0.163 | 0.207 |
| lightgbm | recency_weighted | 0.240 | 0.163 | 0.207 |
| lightgbm | val_weighted | 0.240 | 0.163 | 0.207 |
| random_forest | mean | 0.240 | 0.267 | 0.265 |
| random_forest | recency_weighted | 0.240 | 0.267 | 0.265 |
| random_forest | val_weighted | 0.240 | 0.267 | 0.265 |
| ridge | mean | 0.302 | 0.326 | 0.311 |
| ridge | recency_weighted | 0.302 | 0.326 | 0.311 |
| ridge | val_weighted | 0.302 | 0.326 | 0.311 |
| xgboost | mean | 0.164 | 0.174 | 0.229 |
| xgboost | recency_weighted | 0.164 | 0.174 | 0.229 |
| xgboost | val_weighted | 0.164 | 0.174 | 0.229 |

> 20 yr ≈ 30 yr is expected — the box-score era is data-capped at 1999.

## Market benchmark (§7.4)

_No market benchmark available (run `make benchmark`)._

## NGS-block ablation (§7.3)

_n/a_

## Availability model (§7.4)

Predicting N+1 games played (gates projected eligibility / injury risk):

| model | games MAE | clears-cutoff AUC |
|---|---|---|
| baseline_prior_games | 0.23 | nan |
| gbm_poisson | 1.00 | nan |

## Eligibility-cutoff sensitivity (§6.4)

Best model (`lasso`) Spearman across the candidate games cutoffs (robustness — the headline does not hinge on g\*):

| cutoff (games) | 1 |
|---|---|
| Spearman | 0.311 |


## Data volume

- **Feature rows:** 861 (827 labeled with a next-season target)
- **Features:** 26 columns
- **Seasons:** 1999–2025 (27 seasons)
- **Labeled rows per era:** boxscore 827

## Compute & efficiency

- **Experiment wall-clock:** 12.3s (total model fit time 9.9s across the grid)

Per-model cost vs ranking quality at the headline window (20-yr, g\*=1). **Efficiency** = Spearman per fit-second:

| model | combine | fit (s) | train rows | features | Spearman | P@12 | efficiency |
|---|---|---|---|---|---|---|---|
| lasso | recency_weighted | 0.01 | 636 | 26 | 0.328 | 0.52 | 57.74 |
| lasso | mean | 0.00 | 636 | 26 | 0.328 | 0.52 | 73.46 |
| lasso | val_weighted | 0.01 | 636 | 26 | 0.328 | 0.52 | 24.39 |
| ridge | val_weighted | 0.01 | 636 | 26 | 0.326 | 0.55 | 34.34 |
| ridge | recency_weighted | 0.00 | 636 | 26 | 0.326 | 0.55 | 92.77 |
| ridge | mean | 0.00 | 636 | 26 | 0.326 | 0.55 | 90.85 |
| elasticnet | val_weighted | 0.01 | 636 | 26 | 0.316 | 0.50 | 24.93 |
| elasticnet | recency_weighted | 0.00 | 636 | 26 | 0.316 | 0.50 | 66.79 |
| elasticnet | mean | 0.00 | 636 | 26 | 0.316 | 0.50 | 64.23 |
| random_forest | recency_weighted | 0.19 | 636 | 26 | 0.267 | 0.53 | 1.42 |
| random_forest | mean | 0.19 | 636 | 26 | 0.267 | 0.53 | 1.42 |
| random_forest | val_weighted | 0.38 | 636 | 26 | 0.267 | 0.53 | 0.70 |
| xgboost | mean | 0.17 | 636 | 26 | 0.174 | 0.43 | 1.00 |
| xgboost | val_weighted | 0.36 | 636 | 26 | 0.174 | 0.43 | 0.49 |
| xgboost | recency_weighted | 0.18 | 636 | 26 | 0.174 | 0.43 | 0.95 |
| lightgbm | mean | 0.44 | 636 | 26 | 0.163 | 0.43 | 0.37 |
| lightgbm | val_weighted | 0.83 | 636 | 26 | 0.163 | 0.43 | 0.20 |
| lightgbm | recency_weighted | 0.45 | 636 | 26 | 0.163 | 0.43 | 0.36 |

## Figures

![Spearman by window](figures/report_football_dst_windows.png)

![Model vs market](figures/report_football_dst_vs_market.png)
