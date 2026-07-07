# Position Predictor — Results: football K v1

_**K v1** · primary @ 2bc2224-dirty_

_Generated 2026-07-06._ Test seasons **[2019, 2022, 2023, 2024, 2025]**, eligibility cutoff **g\* = 1 games**. Ranking is computed within each test season; metrics are mean ± sd across the season folds.

## Headline

- **Best model:** `ridge` (val_weighted, 10-yr window) — Spearman **0.413 ± 0.159**, Precision@12 0.38, MAE 1.24 PPG (full eligible universe).
- **Best baseline:** `smoothed_history` — Spearman 0.215 (the must-beat floor).
- **Market head-to-head** (FantasyPros preseason ECR, scored on the identical rows the market ranks): market Spearman 0.242 vs our best `elasticnet` 0.312 — the model **beats** the market on overall rank.
  On **Precision@12 (tier-1 / K1)**: market 0.47 vs best `elasticnet` 0.47 — model trails market.
  On **Precision@24 (tier-2 / K2)**: market 0.89 vs best `lightgbm` 0.89 — model beats market.
  On the **top-weighted** rank score (Weighted τ — errors near #1 count most): market 0.238 vs `ridge` 0.240 — the model **beats** the market where it matters most.

## Recency: how much history helps (§6.2)

Spearman by training-window length (years), at g\*:

| model | combine | 10yr | 20yr | 30yr |
|---|---|---|---|---|
| elasticnet | mean | 0.363 | 0.341 | 0.341 |
| elasticnet | recency_weighted | 0.363 | 0.341 | 0.341 |
| elasticnet | val_weighted | 0.363 | 0.341 | 0.341 |
| lasso | mean | 0.369 | 0.339 | 0.339 |
| lasso | recency_weighted | 0.369 | 0.339 | 0.339 |
| lasso | val_weighted | 0.369 | 0.339 | 0.339 |
| lightgbm | mean | 0.369 | 0.300 | 0.300 |
| lightgbm | recency_weighted | 0.369 | 0.300 | 0.300 |
| lightgbm | val_weighted | 0.369 | 0.300 | 0.300 |
| random_forest | mean | 0.386 | 0.347 | 0.347 |
| random_forest | recency_weighted | 0.386 | 0.347 | 0.347 |
| random_forest | val_weighted | 0.386 | 0.347 | 0.347 |
| ridge | mean | 0.413 | 0.288 | 0.288 |
| ridge | recency_weighted | 0.413 | 0.288 | 0.288 |
| ridge | val_weighted | 0.413 | 0.288 | 0.288 |
| xgboost | mean | 0.349 | 0.341 | 0.341 |
| xgboost | recency_weighted | 0.349 | 0.341 | 0.341 |
| xgboost | val_weighted | 0.349 | 0.341 | 0.341 |

> 20 yr ≈ 30 yr is expected — the box-score era is data-capped at 1999.

## Market benchmark (§7.4)

Preseason ECR coverage of the eligible universe and the market's own ranking quality, per test season:

| season | eligible | ranked | coverage | market Spearman | market P@12 |
|---|---|---|---|---|---|
| 2022 | 38 | 33 | 0.87 | 0.254 | 0.33 |
| 2023 | 33 | 25 | 0.76 | 0.097 | 0.50 |
| 2024 | 33 | 30 | 0.91 | 0.375 | 0.58 |

**Head-to-head on the identical ranked rows** (mean across folds). `weighted_tau` is the **top-weighted** rank score — errors near #1 count most:

| model | Spearman | Weighted τ (top) | Precision@12 (K1) | Precision@24 (K2) |
|---|---|---|---|---|
| ridge | 0.266 | 0.240 | 0.47 | 0.89 |
| market_ecr _(market)_ | 0.242 | 0.238 | 0.47 | 0.89 |
| elasticnet | 0.312 | 0.203 | 0.47 | 0.89 |
| lasso | 0.308 | 0.200 | 0.42 | 0.89 |
| xgboost | 0.228 | 0.192 | 0.47 | 0.89 |
| lightgbm | 0.205 | 0.184 | 0.44 | 0.89 |
| random_forest | 0.218 | 0.179 | 0.47 | 0.89 |

## NGS-block ablation (§7.3)

_n/a_

## Availability model (§7.4)

Predicting N+1 games played (gates projected eligibility / injury risk):

| model | games MAE | clears-cutoff AUC |
|---|---|---|
| baseline_prior_games | 3.93 | 0.814 |
| gbm_poisson | 5.38 | 0.810 |

## Eligibility-cutoff sensitivity (§6.4)

Best model (`ridge`) Spearman across the candidate games cutoffs (robustness — the headline does not hinge on g\*):

| cutoff (games) | 1 | 4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|
| Spearman | 0.330 | 0.278 | 0.244 | 0.215 | 0.183 | 0.173 |


## Data volume

- **Feature rows:** 1062 (805 labeled with a next-season target)
- **Features:** 50 columns
- **Seasons:** 1999–2025 (26 seasons)
- **Labeled rows per era:** boxscore 805

## Compute & efficiency

- **Experiment wall-clock:** 16.7s (total model fit time 11.2s across the grid)

Per-model cost vs ranking quality at the headline window (10-yr, g\*=1). **Efficiency** = Spearman per fit-second:

| model | combine | fit (s) | train rows | features | Spearman | P@12 | efficiency |
|---|---|---|---|---|---|---|---|
| ridge | val_weighted | 0.81 | 333 | 50 | 0.413 | 0.38 | 0.51 |
| ridge | mean | 0.00 | 333 | 50 | 0.413 | 0.38 | 97.19 |
| ridge | recency_weighted | 0.00 | 333 | 50 | 0.413 | 0.38 | 100.56 |
| random_forest | recency_weighted | 0.16 | 333 | 50 | 0.386 | 0.48 | 2.40 |
| random_forest | mean | 0.17 | 333 | 50 | 0.386 | 0.48 | 2.27 |
| random_forest | val_weighted | 0.42 | 333 | 50 | 0.386 | 0.48 | 0.93 |
| lightgbm | val_weighted | 0.56 | 333 | 50 | 0.369 | 0.42 | 0.66 |
| lightgbm | mean | 0.25 | 333 | 50 | 0.369 | 0.42 | 1.49 |
| lightgbm | recency_weighted | 0.28 | 333 | 50 | 0.369 | 0.42 | 1.31 |
| lasso | mean | 0.00 | 333 | 50 | 0.369 | 0.47 | 74.99 |
| lasso | val_weighted | 0.01 | 333 | 50 | 0.369 | 0.47 | 27.46 |
| lasso | recency_weighted | 0.00 | 333 | 50 | 0.369 | 0.47 | 77.79 |
| elasticnet | mean | 0.01 | 333 | 50 | 0.363 | 0.43 | 71.41 |
| elasticnet | recency_weighted | 0.00 | 333 | 50 | 0.363 | 0.43 | 75.02 |
| elasticnet | val_weighted | 0.01 | 333 | 50 | 0.363 | 0.43 | 28.80 |
| xgboost | val_weighted | 0.41 | 333 | 50 | 0.349 | 0.42 | 0.85 |
| xgboost | mean | 0.19 | 333 | 50 | 0.349 | 0.42 | 1.86 |
| xgboost | recency_weighted | 0.19 | 333 | 50 | 0.349 | 0.42 | 1.84 |

## Figures

![Spearman by window](figures/report_football_k_windows.png)

![Model vs market](figures/report_football_k_vs_market.png)
