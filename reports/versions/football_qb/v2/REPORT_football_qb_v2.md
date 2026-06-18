# Position Predictor — Results: football QB v2

_**QB v2** · exclude-covid-2020 @ 7516dfc-dirty_

_Generated 2026-06-17._ Test seasons **[2018, 2019, 2022, 2023, 2024]**, eligibility cutoff **g\* = 7 games**. Ranking is computed within each test season; metrics are mean ± sd across the season folds.

## Headline

- **Best model:** `random_forest` (val_weighted, 20-yr window) — Spearman **0.485 ± 0.217**, Precision@12 0.60, MAE 3.39 PPG (full eligible universe).
- **Best baseline:** `smoothed_history` — Spearman 0.490 (the must-beat floor).
- **Market head-to-head** (FantasyPros preseason ECR, scored on the identical rows the market ranks): market Spearman 0.662 vs our best `xgboost` 0.552 — the model does **not** beat the market on overall rank.
  On **Precision@6 (tier-1 / QB1)**: market 0.56 vs best `elasticnet` 0.56 — model trails market.
  On **Precision@12 (tier-2 / QB2)**: market 0.61 vs best `random_forest` 0.67 — model beats market.
  On the **top-weighted** rank score (Weighted τ — errors near #1 count most): market 0.610 vs `lightgbm` 0.547 — the model does **not** beat the market where it matters most.

## Recency: how much history helps (§6.2)

Spearman by training-window length (years), at g\*:

| model | combine | 10yr | 20yr | 30yr |
|---|---|---|---|---|
| elasticnet | mean | 0.470 | 0.480 | 0.480 |
| elasticnet | recency_weighted | 0.443 | 0.447 | 0.447 |
| elasticnet | val_weighted | 0.471 | 0.479 | 0.479 |
| lasso | mean | 0.457 | 0.479 | 0.479 |
| lasso | recency_weighted | 0.442 | 0.447 | 0.447 |
| lasso | val_weighted | 0.458 | 0.477 | 0.477 |
| lightgbm | mean | 0.447 | 0.440 | 0.440 |
| lightgbm | recency_weighted | 0.424 | 0.425 | 0.425 |
| lightgbm | val_weighted | 0.448 | 0.439 | 0.439 |
| random_forest | mean | 0.446 | 0.485 | 0.485 |
| random_forest | recency_weighted | 0.461 | 0.477 | 0.477 |
| random_forest | val_weighted | 0.447 | 0.485 | 0.485 |
| ridge | mean | 0.460 | 0.474 | 0.474 |
| ridge | recency_weighted | 0.422 | 0.431 | 0.431 |
| ridge | val_weighted | 0.466 | 0.476 | 0.476 |
| xgboost | mean | 0.421 | 0.459 | 0.459 |
| xgboost | recency_weighted | 0.412 | 0.432 | 0.432 |
| xgboost | val_weighted | 0.421 | 0.461 | 0.461 |

> 20 yr ≈ 30 yr is expected — the box-score era is data-capped at 1999.

## Market benchmark (§7.4)

Preseason ECR coverage of the eligible universe and the market's own ranking quality, per test season:

| season | eligible | ranked | coverage | market Spearman | market P@6 |
|---|---|---|---|---|---|
| 2022 | 34 | 31 | 0.91 | 0.600 | 0.67 |
| 2023 | 33 | 30 | 0.91 | 0.696 | 0.50 |
| 2024 | 38 | 37 | 0.97 | 0.690 | 0.50 |

**Head-to-head on the identical ranked rows** (mean across folds). `weighted_tau` is the **top-weighted** rank score — errors near #1 count most:

| model | Spearman | Weighted τ (top) | Precision@6 (QB1) | Precision@12 (QB2) |
|---|---|---|---|---|
| market_ecr _(market)_ | 0.662 | 0.610 | 0.56 | 0.61 |
| lightgbm | 0.521 | 0.547 | 0.44 | 0.67 |
| lasso | 0.499 | 0.531 | 0.56 | 0.56 |
| xgboost | 0.552 | 0.531 | 0.56 | 0.67 |
| elasticnet | 0.503 | 0.528 | 0.56 | 0.58 |
| random_forest | 0.524 | 0.527 | 0.50 | 0.67 |
| ridge | 0.476 | 0.464 | 0.50 | 0.53 |

## NGS-block ablation (§7.3)

`ngs` era model, longest window — **with** NGS: Spearman 0.312, P@12 0.53; **without**: Spearman 0.312, P@12 0.53.
- **Decision:** drop the `ngs_efficiency` block — no top-12 gain out-of-fold, so the ngs era collapses to the snaps schema (coverage flags retained either way).

## Availability model (§7.4)

Predicting N+1 games played (gates projected eligibility / injury risk):

| model | games MAE | clears-cutoff AUC |
|---|---|---|
| baseline_prior_games | 3.71 | 0.859 |
| gbm_poisson | 3.66 | 0.895 |

## Eligibility-cutoff sensitivity (§6.4)

Best model (`random_forest`) Spearman across the candidate games cutoffs (robustness — the headline does not hinge on g\*):

| cutoff (games) | 6 | 7 | 8 | 10 | 12 | 14 |
|---|---|---|---|---|---|---|
| Spearman | 0.507 | 0.472 | 0.433 | 0.386 | 0.358 | 0.333 |


## Data volume

- **Feature rows:** 1918 (1338 labeled with a next-season target)
- **Features:** 73 columns
- **Seasons:** 1999–2024 (25 seasons)
- **Labeled rows per era:** boxscore 763, snaps 224, ngs 351

## Compute & efficiency

- **Experiment wall-clock:** 129.3s (total model fit time 36.3s across the grid)

Per-model cost vs ranking quality at the headline window (20-yr, g\*=7). **Efficiency** = Spearman per fit-second:

| model | combine | fit (s) | train rows | features | Spearman | P@12 | efficiency |
|---|---|---|---|---|---|---|---|
| random_forest | val_weighted | 2.85 | 1044 | 73 | 0.485 | 0.60 | 0.17 |
| random_forest | mean | 1.41 | 1044 | 73 | 0.485 | 0.60 | 0.34 |
| elasticnet | mean | 0.05 | 1044 | 73 | 0.480 | 0.58 | 8.91 |
| elasticnet | val_weighted | 0.12 | 1044 | 73 | 0.479 | 0.60 | 3.90 |
| lasso | mean | 0.05 | 1044 | 73 | 0.479 | 0.60 | 9.23 |
| lasso | val_weighted | 0.12 | 1044 | 73 | 0.477 | 0.60 | 3.98 |
| random_forest | recency_weighted | 1.43 | 1044 | 73 | 0.477 | 0.60 | 0.33 |
| ridge | val_weighted | 0.11 | 1044 | 73 | 0.476 | 0.55 | 4.27 |
| ridge | mean | 0.04 | 1044 | 73 | 0.474 | 0.55 | 11.07 |
| xgboost | val_weighted | 1.96 | 1044 | 73 | 0.461 | 0.60 | 0.24 |
| xgboost | mean | 1.05 | 1044 | 73 | 0.459 | 0.60 | 0.44 |
| lasso | recency_weighted | 0.05 | 1044 | 73 | 0.447 | 0.58 | 8.26 |
| elasticnet | recency_weighted | 0.06 | 1044 | 73 | 0.447 | 0.60 | 7.87 |
| lightgbm | mean | 0.48 | 1044 | 73 | 0.440 | 0.57 | 0.91 |
| lightgbm | val_weighted | 0.84 | 1044 | 73 | 0.439 | 0.57 | 0.53 |
| xgboost | recency_weighted | 1.09 | 1044 | 73 | 0.432 | 0.53 | 0.40 |
| ridge | recency_weighted | 0.05 | 1044 | 73 | 0.431 | 0.55 | 9.32 |
| lightgbm | recency_weighted | 0.43 | 1044 | 73 | 0.425 | 0.57 | 1.00 |

## Figures

![Spearman by window](figures/report_football_qb_windows.png)

![Model vs market](figures/report_football_qb_vs_market.png)
