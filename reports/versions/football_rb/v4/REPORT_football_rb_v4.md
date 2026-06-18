# Position Predictor — Results: football RB v4

_**RB v4** · exclude-covid-2020 @ 7516dfc-dirty_

_Generated 2026-06-17._ Test seasons **[2018, 2019, 2022, 2023, 2024]**, eligibility cutoff **g\* = 4 games**. Ranking is computed within each test season; metrics are mean ± sd across the season folds.

## Headline

- **Best model:** `xgboost` (val_weighted, 20-yr window) — Spearman **0.760 ± 0.023**, Precision@12 0.50, MAE 2.79 PPG (full eligible universe).
- **Best baseline:** `linear` — Spearman 0.746 (the must-beat floor).
- **Market head-to-head** (FantasyPros preseason ECR, scored on the identical rows the market ranks): market Spearman 0.808 vs our best `xgboost` 0.749 — the model does **not** beat the market on overall rank.
  On **Precision@12 (tier-1 / RB1)**: market 0.56 vs best `ridge` 0.58 — model beats market.
  On **Precision@24 (tier-2 / RB2)**: market 0.75 vs best `ridge` 0.75 — model trails market.
  On the **top-weighted** rank score (Weighted τ — errors near #1 count most): market 0.704 vs `ridge` 0.660 — the model does **not** beat the market where it matters most.

## Recency: how much history helps (§6.2)

Spearman by training-window length (years), at g\*:

| model | combine | 10yr | 20yr | 30yr |
|---|---|---|---|---|
| elasticnet | mean | 0.751 | 0.748 | 0.748 |
| elasticnet | recency_weighted | 0.733 | 0.733 | 0.733 |
| elasticnet | val_weighted | 0.751 | 0.748 | 0.748 |
| lasso | mean | 0.753 | 0.751 | 0.751 |
| lasso | recency_weighted | 0.738 | 0.737 | 0.737 |
| lasso | val_weighted | 0.753 | 0.750 | 0.750 |
| lightgbm | mean | 0.749 | 0.751 | 0.751 |
| lightgbm | recency_weighted | 0.736 | 0.736 | 0.736 |
| lightgbm | val_weighted | 0.748 | 0.752 | 0.752 |
| random_forest | mean | 0.752 | 0.757 | 0.757 |
| random_forest | recency_weighted | 0.744 | 0.747 | 0.747 |
| random_forest | val_weighted | 0.752 | 0.757 | 0.757 |
| ridge | mean | 0.750 | 0.750 | 0.750 |
| ridge | recency_weighted | 0.723 | 0.723 | 0.723 |
| ridge | val_weighted | 0.750 | 0.750 | 0.750 |
| xgboost | mean | 0.756 | 0.760 | 0.760 |
| xgboost | recency_weighted | 0.744 | 0.746 | 0.746 |
| xgboost | val_weighted | 0.757 | 0.760 | 0.760 |

> 20 yr ≈ 30 yr is expected — the box-score era is data-capped at 1999.

## Market benchmark (§7.4)

Preseason ECR coverage of the eligible universe and the market's own ranking quality, per test season:

| season | eligible | ranked | coverage | market Spearman | market P@12 |
|---|---|---|---|---|---|
| 2022 | 96 | 89 | 0.93 | 0.818 | 0.58 |
| 2023 | 93 | 81 | 0.87 | 0.803 | 0.42 |
| 2024 | 92 | 91 | 0.99 | 0.804 | 0.67 |

**Head-to-head on the identical ranked rows** (mean across folds). `weighted_tau` is the **top-weighted** rank score — errors near #1 count most:

| model | Spearman | Weighted τ (top) | Precision@12 (RB1) | Precision@24 (RB2) |
|---|---|---|---|---|
| market_ecr _(market)_ | 0.808 | 0.704 | 0.56 | 0.75 |
| ridge | 0.738 | 0.660 | 0.58 | 0.75 |
| random_forest | 0.738 | 0.649 | 0.47 | 0.74 |
| lightgbm | 0.741 | 0.646 | 0.53 | 0.72 |
| lasso | 0.736 | 0.645 | 0.56 | 0.74 |
| xgboost | 0.749 | 0.644 | 0.53 | 0.74 |
| elasticnet | 0.735 | 0.640 | 0.56 | 0.74 |

## NGS-block ablation (§7.3)

`ngs` era model, longest window — **with** NGS: Spearman 0.676, P@12 0.43; **without**: Spearman 0.665, P@12 0.43.
- **Decision:** drop the `ngs_efficiency` block — no top-12 gain out-of-fold, so the ngs era collapses to the snaps schema (coverage flags retained either way).

## Availability model (§7.4)

Predicting N+1 games played (gates projected eligibility / injury risk):

| model | games MAE | clears-cutoff AUC |
|---|---|---|
| baseline_prior_games | 4.29 | 0.782 |
| gbm_poisson | 4.79 | 0.749 |

## Eligibility-cutoff sensitivity (§6.4)

Best model (`xgboost`) Spearman across the candidate games cutoffs (robustness — the headline does not hinge on g\*):

| cutoff (games) | 4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|
| Spearman | 0.754 | 0.746 | 0.726 | 0.686 | 0.654 |


## Data volume

- **Feature rows:** 4034 (2728 labeled with a next-season target)
- **Features:** 81 columns
- **Seasons:** 1999–2024 (25 seasons)
- **Labeled rows per era:** boxscore 1557, snaps 477, ngs 694

## Compute & efficiency

- **Experiment wall-clock:** 268.9s (total model fit time 215.7s across the grid)

Per-model cost vs ranking quality at the headline window (30-yr, g\*=4). **Efficiency** = Spearman per fit-second:

| model | combine | fit (s) | train rows | features | Spearman | P@12 | efficiency |
|---|---|---|---|---|---|---|---|
| xgboost | val_weighted | 2.53 | 2156 | 81 | 0.760 | 0.50 | 0.30 |
| xgboost | mean | 1.70 | 2156 | 81 | 0.760 | 0.50 | 0.45 |
| random_forest | mean | 2.74 | 2156 | 81 | 0.757 | 0.47 | 0.28 |
| random_forest | val_weighted | 5.06 | 2156 | 81 | 0.757 | 0.47 | 0.15 |
| lightgbm | val_weighted | 1.48 | 2156 | 81 | 0.752 | 0.50 | 0.51 |
| lightgbm | mean | 0.86 | 2156 | 81 | 0.751 | 0.50 | 0.88 |
| lasso | mean | 0.07 | 2156 | 81 | 0.751 | 0.53 | 10.17 |
| lasso | val_weighted | 0.15 | 2156 | 81 | 0.750 | 0.53 | 4.87 |
| ridge | mean | 0.07 | 2156 | 81 | 0.750 | 0.58 | 10.55 |
| ridge | val_weighted | 0.14 | 2156 | 81 | 0.750 | 0.57 | 5.37 |
| elasticnet | val_weighted | 0.16 | 2156 | 81 | 0.748 | 0.55 | 4.55 |
| elasticnet | mean | 0.08 | 2156 | 81 | 0.748 | 0.55 | 9.89 |
| random_forest | recency_weighted | 2.75 | 2156 | 81 | 0.747 | 0.45 | 0.27 |
| xgboost | recency_weighted | 1.86 | 2156 | 81 | 0.746 | 0.47 | 0.40 |
| lasso | recency_weighted | 0.08 | 2156 | 81 | 0.737 | 0.55 | 9.42 |
| lightgbm | recency_weighted | 0.86 | 2156 | 81 | 0.736 | 0.52 | 0.85 |
| elasticnet | recency_weighted | 0.08 | 2156 | 81 | 0.733 | 0.53 | 9.06 |
| ridge | recency_weighted | 0.07 | 2156 | 81 | 0.723 | 0.55 | 10.20 |

## Figures

![Spearman by window](figures/report_football_rb_windows.png)

![Model vs market](figures/report_football_rb_vs_market.png)
