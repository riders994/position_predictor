# Position Predictor — Results: football RB v1

_**RB v1** · rb-v3 @ e9b2ee8-dirty_

_Generated 2026-06-17._ Test seasons **[2020, 2021, 2022, 2023, 2024]**, eligibility cutoff **g\* = 4 games**. Ranking is computed within each test season; metrics are mean ± sd across the season folds.

## Headline

- **Best model:** `xgboost` (val_weighted, 20-yr window) — Spearman **0.752 ± 0.031**, Precision@12 0.53, MAE 2.62 PPG (full eligible universe).
- **Best baseline:** `linear` — Spearman 0.720 (the must-beat floor).
- **Market head-to-head** (FantasyPros preseason ECR, scored on the identical rows the market ranks): market Spearman 0.732 vs our best `xgboost` 0.706 — the model does **not** beat the market on overall rank.
  On **Precision@12 (tier-1 / RB1)**: market 0.60 vs best `ridge` 0.65 — model beats market.
  On **Precision@24 (tier-2 / RB2)**: market 0.75 vs best `elasticnet` 0.72 — model trails market.
  On the **top-weighted** rank score (Weighted τ — errors near #1 count most): market 0.689 vs `ridge` 0.663 — the model does **not** beat the market where it matters most.

## Recency: how much history helps (§6.2)

Spearman by training-window length (years), at g\*:

| model | combine | 10yr | 20yr | 30yr |
|---|---|---|---|---|
| elasticnet | mean | 0.741 | 0.746 | 0.746 |
| elasticnet | recency_weighted | 0.744 | 0.749 | 0.749 |
| elasticnet | val_weighted | 0.740 | 0.747 | 0.747 |
| lasso | mean | 0.741 | 0.745 | 0.745 |
| lasso | recency_weighted | 0.745 | 0.748 | 0.748 |
| lasso | val_weighted | 0.741 | 0.744 | 0.744 |
| lightgbm | mean | 0.733 | 0.746 | 0.746 |
| lightgbm | recency_weighted | 0.732 | 0.737 | 0.737 |
| lightgbm | val_weighted | 0.733 | 0.745 | 0.745 |
| random_forest | mean | 0.733 | 0.742 | 0.742 |
| random_forest | recency_weighted | 0.732 | 0.738 | 0.738 |
| random_forest | val_weighted | 0.733 | 0.742 | 0.742 |
| ridge | mean | 0.737 | 0.747 | 0.747 |
| ridge | recency_weighted | 0.745 | 0.749 | 0.749 |
| ridge | val_weighted | 0.738 | 0.747 | 0.747 |
| xgboost | mean | 0.739 | 0.751 | 0.751 |
| xgboost | recency_weighted | 0.738 | 0.746 | 0.746 |
| xgboost | val_weighted | 0.739 | 0.752 | 0.752 |

> 20 yr ≈ 30 yr is expected — the box-score era is data-capped at 1999.

## Market benchmark (§7.4)

Preseason ECR coverage of the eligible universe and the market's own ranking quality, per test season:

| season | eligible | ranked | coverage | market Spearman | market P@12 |
|---|---|---|---|---|---|
| 2020 | 101 | 73 | 0.72 | 0.487 | 0.58 |
| 2021 | 105 | 90 | 0.86 | 0.749 | 0.75 |
| 2022 | 96 | 89 | 0.93 | 0.818 | 0.58 |
| 2023 | 93 | 81 | 0.87 | 0.803 | 0.42 |
| 2024 | 92 | 91 | 0.99 | 0.804 | 0.67 |

**Head-to-head on the identical ranked rows** (mean across folds). `weighted_tau` is the **top-weighted** rank score — errors near #1 count most:

| model | Spearman | Weighted τ (top) | Precision@12 (tier-1) | Precision@24 (tier-2) |
|---|---|---|---|---|
| market_ecr _(market)_ | 0.732 | 0.689 | 0.60 | 0.75 |
| ridge | 0.704 | 0.663 | 0.65 | 0.72 |
| lasso | 0.702 | 0.657 | 0.63 | 0.72 |
| elasticnet | 0.703 | 0.657 | 0.62 | 0.72 |
| xgboost | 0.706 | 0.654 | 0.55 | 0.72 |
| random_forest | 0.691 | 0.650 | 0.55 | 0.71 |
| lightgbm | 0.696 | 0.649 | 0.52 | 0.72 |

## NGS-block ablation (§7.3)

`ngs` era model, longest window — **with** NGS: Spearman 0.711, P@12 0.45; **without**: Spearman 0.708, P@12 0.47.
- **Decision:** drop the `ngs_efficiency` block — no top-12 gain out-of-fold, so the ngs era collapses to the snaps schema (coverage flags retained either way).

## Availability model (§7.4)

Predicting N+1 games played (gates projected eligibility / injury risk):

| model | games MAE | clears-cutoff AUC |
|---|---|---|
| baseline_prior_games | 4.33 | 0.782 |
| gbm_poisson | 4.57 | 0.773 |

## Eligibility-cutoff sensitivity (§6.4)

Best model (`ridge`) Spearman across the candidate games cutoffs (robustness — the headline does not hinge on g\*):

| cutoff (games) | 4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|
| Spearman | 0.745 | 0.748 | 0.731 | 0.714 | 0.714 |


## Data volume (§inputs)

- **Feature rows:** 4200 (2975 labeled with a next-season target)
- **Features:** 80 columns
- **Seasons:** 1999–2024 (26 seasons)
- **Labeled rows per era:** boxscore 1557, snaps 477, ngs 941

## Compute & efficiency (§cost)

- **Experiment wall-clock:** 244.4s (total model fit time 166.3s across the grid)

Per-model cost vs ranking quality at the headline window (30-yr, g\*=4). **Efficiency** = Spearman per fit-second:

| model | combine | fit (s) | train rows | features | Spearman | P@12 | efficiency |
|---|---|---|---|---|---|---|---|
| xgboost | val_weighted | 16.60 | 2380 | 80 | 0.752 | 0.53 | 0.05 |
| xgboost | mean | 8.13 | 2380 | 80 | 0.751 | 0.53 | 0.09 |
| ridge | recency_weighted | 0.11 | 2380 | 80 | 0.749 | 0.62 | 6.70 |
| elasticnet | recency_weighted | 0.11 | 2380 | 80 | 0.749 | 0.58 | 6.57 |
| lasso | recency_weighted | 0.11 | 2380 | 80 | 0.748 | 0.58 | 6.76 |
| ridge | mean | 0.09 | 2380 | 80 | 0.747 | 0.63 | 8.03 |
| ridge | val_weighted | 0.23 | 2380 | 80 | 0.747 | 0.63 | 3.23 |
| elasticnet | val_weighted | 0.27 | 2380 | 80 | 0.747 | 0.60 | 2.72 |
| elasticnet | mean | 0.11 | 2380 | 80 | 0.746 | 0.60 | 6.52 |
| lightgbm | mean | 5.77 | 2380 | 80 | 0.746 | 0.50 | 0.13 |
| xgboost | recency_weighted | 10.52 | 2380 | 80 | 0.746 | 0.50 | 0.07 |
| lightgbm | val_weighted | 12.13 | 2380 | 80 | 0.745 | 0.50 | 0.06 |
| lasso | mean | 0.11 | 2380 | 80 | 0.745 | 0.62 | 7.04 |
| lasso | val_weighted | 0.27 | 2380 | 80 | 0.744 | 0.62 | 2.78 |
| random_forest | val_weighted | 7.52 | 2380 | 80 | 0.742 | 0.53 | 0.10 |
| random_forest | mean | 3.67 | 2380 | 80 | 0.742 | 0.53 | 0.20 |
| random_forest | recency_weighted | 3.84 | 2380 | 80 | 0.738 | 0.53 | 0.19 |
| lightgbm | recency_weighted | 6.86 | 2380 | 80 | 0.737 | 0.53 | 0.11 |

## Figures

![Spearman by window](figures/report_football_rb_windows.png)

![Model vs market](figures/report_football_rb_vs_market.png)
