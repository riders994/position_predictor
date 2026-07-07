# Position Predictor — Results: football QB v2

_**QB v2** · primary @ 2bc2224-dirty_

_Generated 2026-07-06._ Test seasons **[2019, 2022, 2023, 2024, 2025]**, eligibility cutoff **g\* = 7 games**. Ranking is computed within each test season; metrics are mean ± sd across the season folds.

## Headline

- **Best model:** `elasticnet` (mean, 10-yr window) — Spearman **0.569 ± 0.097**, Precision@12 0.65, MAE 4.08 PPG (full eligible universe).
- **Best baseline:** `smoothed_history` — Spearman 0.524 (the must-beat floor).
- **Market head-to-head** (FantasyPros preseason ECR, scored on the identical rows the market ranks): market Spearman 0.661 vs our best `xgboost` 0.529 — the model does **not** beat the market on overall rank.
  On **Precision@6 (tier-1 / QB1)**: market 0.50 vs best `lightgbm` 0.46 — model trails market.
  On **Precision@12 (tier-2 / QB2)**: market 0.62 vs best `random_forest` 0.62 — model trails market.
  On the **top-weighted** rank score (Weighted τ — errors near #1 count most): market 0.520 vs `random_forest` 0.459 — the model does **not** beat the market where it matters most.

## Recency: how much history helps (§6.2)

Spearman by training-window length (years), at g\*:

| model | combine | 10yr | 20yr | 30yr |
|---|---|---|---|---|
| elasticnet | mean | 0.569 | 0.561 | 0.561 |
| elasticnet | recency_weighted | 0.545 | 0.539 | 0.539 |
| elasticnet | val_weighted | 0.568 | 0.561 | 0.561 |
| lasso | mean | 0.559 | 0.547 | 0.547 |
| lasso | recency_weighted | 0.527 | 0.519 | 0.519 |
| lasso | val_weighted | 0.559 | 0.551 | 0.551 |
| lightgbm | mean | 0.527 | 0.533 | 0.533 |
| lightgbm | recency_weighted | 0.503 | 0.508 | 0.508 |
| lightgbm | val_weighted | 0.526 | 0.532 | 0.532 |
| random_forest | mean | 0.517 | 0.542 | 0.542 |
| random_forest | recency_weighted | 0.514 | 0.528 | 0.528 |
| random_forest | val_weighted | 0.515 | 0.542 | 0.542 |
| ridge | mean | 0.551 | 0.548 | 0.548 |
| ridge | recency_weighted | 0.511 | 0.517 | 0.517 |
| ridge | val_weighted | 0.550 | 0.549 | 0.549 |
| xgboost | mean | 0.531 | 0.520 | 0.520 |
| xgboost | recency_weighted | 0.505 | 0.499 | 0.499 |
| xgboost | val_weighted | 0.530 | 0.520 | 0.520 |

> 20 yr ≈ 30 yr is expected — the box-score era is data-capped at 1999.

## Market benchmark (§7.4)

Preseason ECR coverage of the eligible universe and the market's own ranking quality, per test season:

| season | eligible | ranked | coverage | market Spearman | market P@6 |
|---|---|---|---|---|---|
| 2022 | 34 | 31 | 0.91 | 0.608 | 0.67 |
| 2023 | 33 | 30 | 0.91 | 0.741 | 0.50 |
| 2024 | 38 | 37 | 0.97 | 0.661 | 0.50 |
| 2025 | 32 | 31 | 0.97 | 0.633 | 0.33 |

**Head-to-head on the identical ranked rows** (mean across folds). `weighted_tau` is the **top-weighted** rank score — errors near #1 count most:

| model | Spearman | Weighted τ (top) | Precision@6 (QB1) | Precision@12 (QB2) |
|---|---|---|---|---|
| market_ecr _(market)_ | 0.661 | 0.520 | 0.50 | 0.62 |
| random_forest | 0.514 | 0.459 | 0.42 | 0.62 |
| xgboost | 0.529 | 0.452 | 0.46 | 0.56 |
| lightgbm | 0.524 | 0.448 | 0.46 | 0.54 |
| lasso | 0.513 | 0.414 | 0.29 | 0.58 |
| elasticnet | 0.526 | 0.411 | 0.29 | 0.58 |
| ridge | 0.506 | 0.390 | 0.37 | 0.56 |

## NGS-block ablation (§7.3)

`ngs` era model, longest window — **with** NGS: Spearman 0.396, P@12 0.47; **without**: Spearman 0.396, P@12 0.47.
- **Decision:** drop the `ngs_efficiency` block — no top-12 gain out-of-fold, so the ngs era collapses to the snaps schema (coverage flags retained either way).

## Availability model (§7.4)

Predicting N+1 games played (gates projected eligibility / injury risk):

| model | games MAE | clears-cutoff AUC |
|---|---|---|
| baseline_prior_games | 3.75 | 0.859 |
| gbm_poisson | 3.65 | 0.880 |

## Eligibility-cutoff sensitivity (§6.4)

Best model (`elasticnet`) Spearman across the candidate games cutoffs (robustness — the headline does not hinge on g\*):

| cutoff (games) | 6 | 7 | 8 | 10 | 12 | 14 |
|---|---|---|---|---|---|---|
| Spearman | 0.607 | 0.556 | 0.547 | 0.485 | 0.446 | 0.372 |


## Data volume

- **Feature rows:** 1996 (1398 labeled with a next-season target)
- **Features:** 73 columns
- **Seasons:** 1999–2025 (26 seasons)
- **Labeled rows per era:** boxscore 762, snaps 222, ngs 414

## Compute & efficiency

- **Experiment wall-clock:** 37.5s (total model fit time 25.4s across the grid)

Per-model cost vs ranking quality at the headline window (10-yr, g\*=7). **Efficiency** = Spearman per fit-second:

| model | combine | fit (s) | train rows | features | Spearman | P@12 | efficiency |
|---|---|---|---|---|---|---|---|
| elasticnet | mean | 0.02 | 556 | 73 | 0.569 | 0.65 | 32.37 |
| elasticnet | val_weighted | 0.04 | 556 | 73 | 0.568 | 0.65 | 14.52 |
| lasso | mean | 0.02 | 556 | 73 | 0.559 | 0.63 | 27.65 |
| lasso | val_weighted | 0.04 | 556 | 73 | 0.559 | 0.63 | 12.53 |
| ridge | mean | 0.01 | 556 | 73 | 0.551 | 0.62 | 42.40 |
| ridge | val_weighted | 0.80 | 556 | 73 | 0.550 | 0.62 | 0.69 |
| elasticnet | recency_weighted | 0.02 | 556 | 73 | 0.545 | 0.60 | 30.76 |
| xgboost | mean | 0.69 | 556 | 73 | 0.531 | 0.62 | 0.77 |
| xgboost | val_weighted | 1.14 | 556 | 73 | 0.530 | 0.62 | 0.46 |
| lasso | recency_weighted | 0.02 | 556 | 73 | 0.527 | 0.62 | 26.26 |
| lightgbm | mean | 0.56 | 556 | 73 | 0.527 | 0.58 | 0.94 |
| lightgbm | val_weighted | 0.88 | 556 | 73 | 0.526 | 0.58 | 0.60 |
| random_forest | mean | 0.41 | 556 | 73 | 0.517 | 0.58 | 1.25 |
| random_forest | val_weighted | 0.86 | 556 | 73 | 0.515 | 0.58 | 0.60 |
| random_forest | recency_weighted | 0.40 | 556 | 73 | 0.514 | 0.58 | 1.27 |
| ridge | recency_weighted | 0.01 | 556 | 73 | 0.511 | 0.58 | 39.94 |
| xgboost | recency_weighted | 0.63 | 556 | 73 | 0.505 | 0.57 | 0.80 |
| lightgbm | recency_weighted | 0.46 | 556 | 73 | 0.503 | 0.55 | 1.09 |

## Figures

![Spearman by window](figures/report_football_qb_windows.png)

![Model vs market](figures/report_football_qb_vs_market.png)
