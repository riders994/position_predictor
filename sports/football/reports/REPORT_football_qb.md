# Position Predictor — Results: football QB v2

_**QB v2** · migrate-nflreadpy @ 8c77d9f-dirty_

_Generated 2026-06-23._ Test seasons **[2019, 2022, 2023, 2024, 2025]**, eligibility cutoff **g\* = 7 games**. Ranking is computed within each test season; metrics are mean ± sd across the season folds.

## Headline

- **Best model:** `lasso` (val_weighted, 20-yr window) — Spearman **0.572 ± 0.082**, Precision@12 0.62, MAE 3.00 PPG (full eligible universe).
- **Best baseline:** `smoothed_history` — Spearman 0.529 (the must-beat floor).
- **Market head-to-head** (FantasyPros preseason ECR, scored on the identical rows the market ranks): market Spearman 0.659 vs our best `xgboost` 0.537 — the model does **not** beat the market on overall rank.
  On **Precision@6 (tier-1 / QB1)**: market 0.50 vs best `xgboost` 0.54 — model beats market.
  On **Precision@12 (tier-2 / QB2)**: market 0.60 vs best `elasticnet` 0.62 — model beats market.
  On the **top-weighted** rank score (Weighted τ — errors near #1 count most): market 0.561 vs `random_forest` 0.525 — the model does **not** beat the market where it matters most.

## Recency: how much history helps (§6.2)

Spearman by training-window length (years), at g\*:

| model | combine | 10yr | 20yr | 30yr |
|---|---|---|---|---|
| elasticnet | mean | 0.548 | 0.570 | 0.570 |
| elasticnet | recency_weighted | 0.533 | 0.543 | 0.543 |
| elasticnet | val_weighted | 0.546 | 0.570 | 0.570 |
| lasso | mean | 0.551 | 0.571 | 0.571 |
| lasso | recency_weighted | 0.534 | 0.538 | 0.538 |
| lasso | val_weighted | 0.553 | 0.572 | 0.572 |
| lightgbm | mean | 0.533 | 0.543 | 0.543 |
| lightgbm | recency_weighted | 0.505 | 0.515 | 0.515 |
| lightgbm | val_weighted | 0.534 | 0.544 | 0.544 |
| random_forest | mean | 0.523 | 0.547 | 0.547 |
| random_forest | recency_weighted | 0.521 | 0.530 | 0.530 |
| random_forest | val_weighted | 0.521 | 0.546 | 0.546 |
| ridge | mean | 0.526 | 0.557 | 0.557 |
| ridge | recency_weighted | 0.509 | 0.509 | 0.509 |
| ridge | val_weighted | 0.527 | 0.557 | 0.557 |
| xgboost | mean | 0.531 | 0.550 | 0.550 |
| xgboost | recency_weighted | 0.505 | 0.516 | 0.516 |
| xgboost | val_weighted | 0.531 | 0.549 | 0.549 |

> 20 yr ≈ 30 yr is expected — the box-score era is data-capped at 1999.

## Market benchmark (§7.4)

Preseason ECR coverage of the eligible universe and the market's own ranking quality, per test season:

| season | eligible | ranked | coverage | market Spearman | market P@6 |
|---|---|---|---|---|---|
| 2022 | 34 | 31 | 0.91 | 0.600 | 0.67 |
| 2023 | 33 | 30 | 0.91 | 0.696 | 0.50 |
| 2024 | 38 | 37 | 0.97 | 0.691 | 0.50 |
| 2025 | 32 | 31 | 0.97 | 0.652 | 0.33 |

**Head-to-head on the identical ranked rows** (mean across folds). `weighted_tau` is the **top-weighted** rank score — errors near #1 count most:

| model | Spearman | Weighted τ (top) | Precision@6 (QB1) | Precision@12 (QB2) |
|---|---|---|---|---|
| market_ecr _(market)_ | 0.659 | 0.561 | 0.50 | 0.60 |
| random_forest | 0.518 | 0.525 | 0.46 | 0.58 |
| xgboost | 0.537 | 0.499 | 0.54 | 0.62 |
| lightgbm | 0.522 | 0.484 | 0.46 | 0.62 |
| lasso | 0.528 | 0.431 | 0.37 | 0.60 |
| elasticnet | 0.528 | 0.428 | 0.37 | 0.62 |
| ridge | 0.511 | 0.398 | 0.37 | 0.56 |

## NGS-block ablation (§7.3)

`ngs` era model, longest window — **with** NGS: Spearman 0.334, P@12 0.52; **without**: Spearman 0.334, P@12 0.52.
- **Decision:** drop the `ngs_efficiency` block — no top-12 gain out-of-fold, so the ngs era collapses to the snaps schema (coverage flags retained either way).

## Availability model (§7.4)

Predicting N+1 games played (gates projected eligibility / injury risk):

| model | games MAE | clears-cutoff AUC |
|---|---|---|
| baseline_prior_games | 3.75 | 0.859 |
| gbm_poisson | 3.59 | 0.887 |

## Eligibility-cutoff sensitivity (§6.4)

Best model (`elasticnet`) Spearman across the candidate games cutoffs (robustness — the headline does not hinge on g\*):

| cutoff (games) | 6 | 7 | 8 | 10 | 12 | 14 |
|---|---|---|---|---|---|---|
| Spearman | 0.606 | 0.555 | 0.542 | 0.486 | 0.450 | 0.397 |


## Data volume

- **Feature rows:** 1996 (1398 labeled with a next-season target)
- **Features:** 73 columns
- **Seasons:** 1999–2025 (26 seasons)
- **Labeled rows per era:** boxscore 762, snaps 222, ngs 414

## Compute & efficiency

- **Experiment wall-clock:** 65.3s (total model fit time 35.5s across the grid)

Per-model cost vs ranking quality at the headline window (30-yr, g\*=7). **Efficiency** = Spearman per fit-second:

| model | combine | fit (s) | train rows | features | Spearman | P@12 | efficiency |
|---|---|---|---|---|---|---|---|
| lasso | val_weighted | 0.11 | 1092 | 73 | 0.572 | 0.62 | 5.17 |
| lasso | mean | 0.05 | 1092 | 73 | 0.571 | 0.62 | 11.52 |
| elasticnet | val_weighted | 0.11 | 1092 | 73 | 0.570 | 0.63 | 5.18 |
| elasticnet | mean | 0.05 | 1092 | 73 | 0.570 | 0.62 | 11.36 |
| ridge | val_weighted | 0.10 | 1092 | 73 | 0.557 | 0.57 | 5.59 |
| ridge | mean | 0.04 | 1092 | 73 | 0.557 | 0.57 | 12.73 |
| xgboost | mean | 1.13 | 1092 | 73 | 0.550 | 0.62 | 0.49 |
| xgboost | val_weighted | 1.86 | 1092 | 73 | 0.549 | 0.62 | 0.30 |
| random_forest | mean | 1.40 | 1092 | 73 | 0.547 | 0.58 | 0.39 |
| random_forest | val_weighted | 3.08 | 1092 | 73 | 0.546 | 0.58 | 0.18 |
| lightgbm | val_weighted | 0.83 | 1092 | 73 | 0.544 | 0.62 | 0.66 |
| lightgbm | mean | 0.46 | 1092 | 73 | 0.543 | 0.62 | 1.19 |
| elasticnet | recency_weighted | 0.05 | 1092 | 73 | 0.543 | 0.58 | 10.74 |
| lasso | recency_weighted | 0.05 | 1092 | 73 | 0.538 | 0.58 | 11.90 |
| random_forest | recency_weighted | 1.41 | 1092 | 73 | 0.530 | 0.62 | 0.38 |
| xgboost | recency_weighted | 1.15 | 1092 | 73 | 0.516 | 0.62 | 0.45 |
| lightgbm | recency_weighted | 0.43 | 1092 | 73 | 0.515 | 0.60 | 1.19 |
| ridge | recency_weighted | 0.04 | 1092 | 73 | 0.509 | 0.57 | 11.57 |

## Figures

![Spearman by window](figures/report_football_qb_windows.png)

![Model vs market](figures/report_football_qb_vs_market.png)
