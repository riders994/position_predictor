# Position Predictor — Results: football TE v1

_**TE v1** · primary @ 2bc2224-dirty_

_Generated 2026-07-06._ Test seasons **[2019, 2022, 2023, 2024, 2025]**, eligibility cutoff **g\* = 5 games**. Ranking is computed within each test season; metrics are mean ± sd across the season folds.

## Headline

- **Best model:** `lasso` (val_weighted, 20-yr window) — Spearman **0.744 ± 0.071**, Precision@12 0.58, MAE 1.89 PPG (full eligible universe).
- **Best baseline:** `linear` — Spearman 0.727 (the must-beat floor).
- **Market head-to-head** (FantasyPros preseason ECR, scored on the identical rows the market ranks): market Spearman 0.780 vs our best `lasso` 0.723 — the model does **not** beat the market on overall rank.
  On **Precision@12 (tier-1 / TE1)**: market 0.69 vs best `xgboost` 0.67 — model trails market.
  On **Precision@24 (tier-2 / TE2)**: market 0.80 vs best `random_forest` 0.80 — model beats market.
  On the **top-weighted** rank score (Weighted τ — errors near #1 count most): market 0.751 vs `lasso` 0.721 — the model does **not** beat the market where it matters most.

## Recency: how much history helps (§6.2)

Spearman by training-window length (years), at g\*:

| model | combine | 10yr | 20yr | 30yr |
|---|---|---|---|---|
| elasticnet | mean | 0.717 | 0.731 | 0.731 |
| elasticnet | recency_weighted | 0.701 | 0.706 | 0.706 |
| elasticnet | val_weighted | 0.718 | 0.732 | 0.732 |
| lasso | mean | 0.737 | 0.744 | 0.744 |
| lasso | recency_weighted | 0.732 | 0.734 | 0.734 |
| lasso | val_weighted | 0.737 | 0.744 | 0.744 |
| lightgbm | mean | 0.728 | 0.734 | 0.734 |
| lightgbm | recency_weighted | 0.728 | 0.730 | 0.730 |
| lightgbm | val_weighted | 0.728 | 0.735 | 0.735 |
| random_forest | mean | 0.742 | 0.739 | 0.739 |
| random_forest | recency_weighted | 0.739 | 0.738 | 0.738 |
| random_forest | val_weighted | 0.742 | 0.739 | 0.739 |
| ridge | mean | 0.648 | 0.650 | 0.650 |
| ridge | recency_weighted | 0.627 | 0.630 | 0.630 |
| ridge | val_weighted | 0.647 | 0.653 | 0.653 |
| xgboost | mean | 0.733 | 0.738 | 0.738 |
| xgboost | recency_weighted | 0.733 | 0.731 | 0.731 |
| xgboost | val_weighted | 0.734 | 0.738 | 0.738 |

> 20 yr ≈ 30 yr is expected — the box-score era is data-capped at 1999.

## Market benchmark (§7.4)

Preseason ECR coverage of the eligible universe and the market's own ranking quality, per test season:

| season | eligible | ranked | coverage | market Spearman | market P@12 |
|---|---|---|---|---|---|
| 2022 | 99 | 72 | 0.73 | 0.773 | 0.75 |
| 2023 | 96 | 46 | 0.48 | 0.773 | 0.67 |
| 2024 | 96 | 90 | 0.94 | 0.771 | 0.67 |
| 2025 | 107 | 60 | 0.56 | 0.803 | 0.67 |

**Head-to-head on the identical ranked rows** (mean across folds). `weighted_tau` is the **top-weighted** rank score — errors near #1 count most:

| model | Spearman | Weighted τ (top) | Precision@12 (TE1) | Precision@24 (TE2) |
|---|---|---|---|---|
| market_ecr _(market)_ | 0.780 | 0.751 | 0.69 | 0.80 |
| lasso | 0.723 | 0.721 | 0.60 | 0.78 |
| random_forest | 0.722 | 0.719 | 0.62 | 0.80 |
| elasticnet | 0.707 | 0.717 | 0.58 | 0.78 |
| xgboost | 0.710 | 0.715 | 0.67 | 0.76 |
| lightgbm | 0.717 | 0.704 | 0.60 | 0.77 |
| ridge | 0.627 | 0.659 | 0.58 | 0.72 |

## NGS-block ablation (§7.3)

`ngs` era model, longest window — **with** NGS: Spearman 0.701, P@12 0.58; **without**: Spearman 0.697, P@12 0.62.
- **Decision:** drop the `ngs_efficiency` block — no top-12 gain out-of-fold, so the ngs era collapses to the snaps schema (coverage flags retained either way).

## Availability model (§7.4)

Predicting N+1 games played (gates projected eligibility / injury risk):

| model | games MAE | clears-cutoff AUC |
|---|---|---|
| baseline_prior_games | 4.03 | 0.799 |
| gbm_poisson | 4.44 | 0.795 |

## Eligibility-cutoff sensitivity (§6.4)

Best model (`random_forest`) Spearman across the candidate games cutoffs (robustness — the headline does not hinge on g\*):

| cutoff (games) | 4 | 5 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|
| Spearman | 0.747 | 0.740 | 0.742 | 0.739 | 0.737 | 0.731 |


## Data volume

- **Feature rows:** 3027 (2079 labeled with a next-season target)
- **Features:** 81 columns
- **Seasons:** 1999–2025 (26 seasons)
- **Labeled rows per era:** boxscore 1058, snaps 350, ngs 671

## Compute & efficiency

- **Experiment wall-clock:** 48.1s (total model fit time 32.7s across the grid)

Per-model cost vs ranking quality at the headline window (20-yr, g\*=5). **Efficiency** = Spearman per fit-second:

| model | combine | fit (s) | train rows | features | Spearman | P@12 | efficiency |
|---|---|---|---|---|---|---|---|
| lasso | val_weighted | 0.11 | 1588 | 81 | 0.744 | 0.58 | 6.71 |
| lasso | mean | 0.07 | 1588 | 81 | 0.744 | 0.58 | 10.65 |
| random_forest | mean | 0.81 | 1588 | 81 | 0.739 | 0.62 | 0.91 |
| random_forest | val_weighted | 1.64 | 1588 | 81 | 0.739 | 0.62 | 0.45 |
| xgboost | mean | 0.75 | 1588 | 81 | 0.738 | 0.65 | 0.98 |
| xgboost | val_weighted | 1.20 | 1588 | 81 | 0.738 | 0.65 | 0.62 |
| random_forest | recency_weighted | 0.83 | 1588 | 81 | 0.738 | 0.60 | 0.89 |
| lightgbm | val_weighted | 2.42 | 1588 | 81 | 0.735 | 0.60 | 0.30 |
| lasso | recency_weighted | 0.17 | 1588 | 81 | 0.734 | 0.58 | 4.25 |
| lightgbm | mean | 1.29 | 1588 | 81 | 0.734 | 0.60 | 0.57 |
| elasticnet | val_weighted | 0.06 | 1588 | 81 | 0.732 | 0.58 | 11.62 |
| xgboost | recency_weighted | 0.79 | 1588 | 81 | 0.731 | 0.62 | 0.93 |
| elasticnet | mean | 0.03 | 1588 | 81 | 0.731 | 0.60 | 23.41 |
| lightgbm | recency_weighted | 1.71 | 1588 | 81 | 0.730 | 0.60 | 0.43 |
| elasticnet | recency_weighted | 0.04 | 1588 | 81 | 0.706 | 0.60 | 18.68 |
| ridge | val_weighted | 0.04 | 1588 | 81 | 0.653 | 0.57 | 16.26 |
| ridge | mean | 0.02 | 1588 | 81 | 0.650 | 0.57 | 37.24 |
| ridge | recency_weighted | 0.02 | 1588 | 81 | 0.630 | 0.57 | 31.91 |

## Figures

![Spearman by window](figures/report_football_te_windows.png)

![Model vs market](figures/report_football_te_vs_market.png)
