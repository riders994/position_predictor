# Position Predictor — Results: football WR v1

_**WR v1** · rb-v3 @ e9b2ee8-dirty_

_Generated 2026-06-17._ Test seasons **[2020, 2021, 2022, 2023, 2024]**, eligibility cutoff **g\* = 7 games**. Ranking is computed within each test season; metrics are mean ± sd across the season folds.

## Headline

- **Best model:** `ridge` (val_weighted, 20-yr window) — Spearman **0.750 ± 0.034**, Precision@12 0.50, MAE 2.49 PPG (full eligible universe).
- **Best baseline:** `linear` — Spearman 0.732 (the must-beat floor).
- **Market head-to-head** (FantasyPros preseason ECR, scored on the identical rows the market ranks): market Spearman 0.767 vs our best `ridge` 0.748 — the model does **not** beat the market on overall rank.
  On **Precision@12 (tier-1 / RB1)**: market 0.57 vs best `xgboost` 0.58 — model beats market.
  On **Precision@24 (tier-2 / RB2)**: market 0.65 vs best `ridge` 0.69 — model beats market.
  On the **top-weighted** rank score (Weighted τ — errors near #1 count most): market 0.733 vs `elasticnet` 0.712 — the model does **not** beat the market where it matters most.

## Recency: how much history helps (§6.2)

Spearman by training-window length (years), at g\*:

| model | combine | 10yr | 20yr | 30yr |
|---|---|---|---|---|
| elasticnet | mean | 0.746 | 0.748 | 0.748 |
| elasticnet | recency_weighted | 0.747 | 0.749 | 0.749 |
| elasticnet | val_weighted | 0.746 | 0.748 | 0.748 |
| lasso | mean | 0.745 | 0.748 | 0.748 |
| lasso | recency_weighted | 0.747 | 0.746 | 0.746 |
| lasso | val_weighted | 0.745 | 0.748 | 0.748 |
| lightgbm | mean | 0.713 | 0.720 | 0.720 |
| lightgbm | recency_weighted | 0.712 | 0.714 | 0.714 |
| lightgbm | val_weighted | 0.714 | 0.720 | 0.720 |
| random_forest | mean | 0.717 | 0.722 | 0.722 |
| random_forest | recency_weighted | 0.714 | 0.717 | 0.717 |
| random_forest | val_weighted | 0.717 | 0.722 | 0.722 |
| ridge | mean | 0.745 | 0.750 | 0.750 |
| ridge | recency_weighted | 0.746 | 0.749 | 0.749 |
| ridge | val_weighted | 0.745 | 0.750 | 0.750 |
| xgboost | mean | 0.719 | 0.723 | 0.723 |
| xgboost | recency_weighted | 0.719 | 0.721 | 0.721 |
| xgboost | val_weighted | 0.719 | 0.723 | 0.723 |

> 20 yr ≈ 30 yr is expected — the box-score era is data-capped at 1999.

## Market benchmark (§7.4)

Preseason ECR coverage of the eligible universe and the market's own ranking quality, per test season:

| season | eligible | ranked | coverage | market Spearman | market P@12 |
|---|---|---|---|---|---|
| 2020 | 157 | 102 | 0.65 | 0.728 | 0.58 |
| 2021 | 172 | 108 | 0.63 | 0.733 | 0.42 |
| 2022 | 169 | 143 | 0.85 | 0.810 | 0.75 |
| 2023 | 153 | 109 | 0.71 | 0.784 | 0.50 |
| 2024 | 163 | 157 | 0.96 | 0.781 | 0.58 |

**Head-to-head on the identical ranked rows** (mean across folds). `weighted_tau` is the **top-weighted** rank score — errors near #1 count most:

| model | Spearman | Weighted τ (top) | Precision@12 (tier-1) | Precision@24 (tier-2) |
|---|---|---|---|---|
| market_ecr _(market)_ | 0.767 | 0.733 | 0.57 | 0.65 |
| elasticnet | 0.748 | 0.712 | 0.52 | 0.67 |
| ridge | 0.748 | 0.712 | 0.52 | 0.69 |
| lasso | 0.747 | 0.712 | 0.52 | 0.67 |
| xgboost | 0.724 | 0.708 | 0.58 | 0.67 |
| lightgbm | 0.720 | 0.707 | 0.55 | 0.67 |
| random_forest | 0.716 | 0.699 | 0.53 | 0.67 |

## NGS-block ablation (§7.3)

`ngs` era model, longest window — **with** NGS: Spearman 0.689, P@12 0.53; **without**: Spearman 0.691, P@12 0.48.
- **Decision:** keep the `ngs_efficiency` block — it improves (coverage flags retained either way).

## Availability model (§7.4)

Predicting N+1 games played (gates projected eligibility / injury risk):

| model | games MAE | clears-cutoff AUC |
|---|---|---|
| baseline_prior_games | 4.26 | 0.822 |
| gbm_poisson | 4.53 | 0.834 |

## Eligibility-cutoff sensitivity (§6.4)

Best model (`elasticnet`) Spearman across the candidate games cutoffs (robustness — the headline does not hinge on g\*):

| cutoff (games) | 4 | 6 | 7 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|
| Spearman | 0.769 | 0.768 | 0.748 | 0.776 | 0.777 | 0.778 |


## Data volume

- **Feature rows:** 5051 (3532 labeled with a next-season target)
- **Features:** 81 columns
- **Seasons:** 1999–2024 (26 seasons)
- **Labeled rows per era:** boxscore 1720, snaps 557, ngs 1255

## Compute & efficiency

- **Experiment wall-clock:** 88.6s (total model fit time 55.5s across the grid)

Per-model cost vs ranking quality at the headline window (30-yr, g\*=7). **Efficiency** = Spearman per fit-second:

| model | combine | fit (s) | train rows | features | Spearman | P@12 | efficiency |
|---|---|---|---|---|---|---|---|
| ridge | val_weighted | 0.13 | 2718 | 81 | 0.750 | 0.50 | 5.62 |
| ridge | mean | 0.07 | 2718 | 81 | 0.750 | 0.50 | 10.86 |
| elasticnet | recency_weighted | 0.08 | 2718 | 81 | 0.749 | 0.52 | 9.29 |
| ridge | recency_weighted | 0.07 | 2718 | 81 | 0.749 | 0.50 | 11.40 |
| lasso | val_weighted | 0.18 | 2718 | 81 | 0.748 | 0.50 | 4.11 |
| lasso | mean | 0.10 | 2718 | 81 | 0.748 | 0.50 | 7.80 |
| elasticnet | mean | 0.08 | 2718 | 81 | 0.748 | 0.50 | 9.59 |
| elasticnet | val_weighted | 0.17 | 2718 | 81 | 0.748 | 0.50 | 4.49 |
| lasso | recency_weighted | 0.09 | 2718 | 81 | 0.746 | 0.55 | 8.02 |
| xgboost | val_weighted | 2.55 | 2718 | 81 | 0.723 | 0.55 | 0.28 |
| xgboost | mean | 1.46 | 2718 | 81 | 0.723 | 0.55 | 0.50 |
| random_forest | mean | 2.68 | 2718 | 81 | 0.722 | 0.52 | 0.27 |
| random_forest | val_weighted | 5.23 | 2718 | 81 | 0.722 | 0.52 | 0.14 |
| xgboost | recency_weighted | 1.47 | 2718 | 81 | 0.721 | 0.55 | 0.49 |
| lightgbm | mean | 0.92 | 2718 | 81 | 0.720 | 0.53 | 0.78 |
| lightgbm | val_weighted | 1.77 | 2718 | 81 | 0.720 | 0.53 | 0.41 |
| random_forest | recency_weighted | 2.68 | 2718 | 81 | 0.717 | 0.52 | 0.27 |
| lightgbm | recency_weighted | 0.81 | 2718 | 81 | 0.714 | 0.52 | 0.88 |

## Figures

![Spearman by window](figures/report_football_wr_windows.png)

![Model vs market](figures/report_football_wr_vs_market.png)
