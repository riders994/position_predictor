# Position Predictor — Results: football WR v2

_**WR v2** · exclude-covid-2020 @ 7516dfc-dirty_

_Generated 2026-06-17._ Test seasons **[2018, 2019, 2022, 2023, 2024]**, eligibility cutoff **g\* = 7 games**. Ranking is computed within each test season; metrics are mean ± sd across the season folds.

## Headline

- **Best model:** `lasso` (val_weighted, 20-yr window) — Spearman **0.730 ± 0.034**, Precision@12 0.60, MAE 2.70 PPG (full eligible universe).
- **Best baseline:** `linear` — Spearman 0.718 (the must-beat floor).
- **Market head-to-head** (FantasyPros preseason ECR, scored on the identical rows the market ranks): market Spearman 0.792 vs our best `lasso` 0.745 — the model does **not** beat the market on overall rank.
  On **Precision@12 (tier-1 / WR1)**: market 0.61 vs best `lightgbm` 0.58 — model trails market.
  On **Precision@24 (tier-2 / WR2)**: market 0.65 vs best `ridge` 0.71 — model beats market.
  On the **top-weighted** rank score (Weighted τ — errors near #1 count most): market 0.776 vs `lightgbm` 0.752 — the model does **not** beat the market where it matters most.

## Recency: how much history helps (§6.2)

Spearman by training-window length (years), at g\*:

| model | combine | 10yr | 20yr | 30yr |
|---|---|---|---|---|
| elasticnet | mean | 0.727 | 0.730 | 0.730 |
| elasticnet | recency_weighted | 0.721 | 0.722 | 0.722 |
| elasticnet | val_weighted | 0.727 | 0.730 | 0.730 |
| lasso | mean | 0.728 | 0.730 | 0.730 |
| lasso | recency_weighted | 0.724 | 0.723 | 0.723 |
| lasso | val_weighted | 0.729 | 0.730 | 0.730 |
| lightgbm | mean | 0.712 | 0.715 | 0.715 |
| lightgbm | recency_weighted | 0.704 | 0.705 | 0.705 |
| lightgbm | val_weighted | 0.712 | 0.714 | 0.714 |
| random_forest | mean | 0.707 | 0.713 | 0.713 |
| random_forest | recency_weighted | 0.704 | 0.707 | 0.707 |
| random_forest | val_weighted | 0.707 | 0.713 | 0.713 |
| ridge | mean | 0.718 | 0.723 | 0.723 |
| ridge | recency_weighted | 0.702 | 0.704 | 0.704 |
| ridge | val_weighted | 0.718 | 0.724 | 0.724 |
| xgboost | mean | 0.714 | 0.716 | 0.716 |
| xgboost | recency_weighted | 0.703 | 0.704 | 0.704 |
| xgboost | val_weighted | 0.714 | 0.717 | 0.717 |

> 20 yr ≈ 30 yr is expected — the box-score era is data-capped at 1999.

## Market benchmark (§7.4)

Preseason ECR coverage of the eligible universe and the market's own ranking quality, per test season:

| season | eligible | ranked | coverage | market Spearman | market P@12 |
|---|---|---|---|---|---|
| 2022 | 169 | 143 | 0.85 | 0.810 | 0.75 |
| 2023 | 153 | 109 | 0.71 | 0.784 | 0.50 |
| 2024 | 163 | 157 | 0.96 | 0.781 | 0.58 |

**Head-to-head on the identical ranked rows** (mean across folds). `weighted_tau` is the **top-weighted** rank score — errors near #1 count most:

| model | Spearman | Weighted τ (top) | Precision@12 (WR1) | Precision@24 (WR2) |
|---|---|---|---|---|
| market_ecr _(market)_ | 0.792 | 0.776 | 0.61 | 0.65 |
| lightgbm | 0.723 | 0.752 | 0.58 | 0.67 |
| ridge | 0.736 | 0.748 | 0.56 | 0.71 |
| lasso | 0.745 | 0.747 | 0.56 | 0.69 |
| elasticnet | 0.744 | 0.746 | 0.56 | 0.67 |
| xgboost | 0.722 | 0.745 | 0.56 | 0.68 |
| random_forest | 0.715 | 0.743 | 0.56 | 0.68 |

## NGS-block ablation (§7.3)

`ngs` era model, longest window — **with** NGS: Spearman 0.662, P@12 0.48; **without**: Spearman 0.638, P@12 0.43.
- **Decision:** keep the `ngs_efficiency` block — it improves (coverage flags retained either way).

## Availability model (§7.4)

Predicting N+1 games played (gates projected eligibility / injury risk):

| model | games MAE | clears-cutoff AUC |
|---|---|---|
| baseline_prior_games | 4.33 | 0.825 |
| gbm_poisson | 4.66 | 0.834 |

## Eligibility-cutoff sensitivity (§6.4)

Best model (`elasticnet`) Spearman across the candidate games cutoffs (robustness — the headline does not hinge on g\*):

| cutoff (games) | 4 | 6 | 7 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|
| Spearman | 0.724 | 0.729 | 0.727 | 0.740 | 0.746 | 0.752 |


## Data volume

- **Feature rows:** 4831 (3203 labeled with a next-season target)
- **Features:** 81 columns
- **Seasons:** 1999–2024 (25 seasons)
- **Labeled rows per era:** boxscore 1720, snaps 557, ngs 926

## Compute & efficiency

- **Experiment wall-clock:** 93.0s (total model fit time 59.2s across the grid)

Per-model cost vs ranking quality at the headline window (30-yr, g\*=7). **Efficiency** = Spearman per fit-second:

| model | combine | fit (s) | train rows | features | Spearman | P@12 | efficiency |
|---|---|---|---|---|---|---|---|
| lasso | val_weighted | 0.18 | 2423 | 81 | 0.730 | 0.60 | 4.11 |
| lasso | mean | 0.09 | 2423 | 81 | 0.730 | 0.60 | 8.32 |
| elasticnet | val_weighted | 0.17 | 2423 | 81 | 0.730 | 0.60 | 4.29 |
| elasticnet | mean | 0.08 | 2423 | 81 | 0.730 | 0.60 | 8.94 |
| ridge | val_weighted | 0.16 | 2423 | 81 | 0.724 | 0.60 | 4.62 |
| lasso | recency_weighted | 0.08 | 2423 | 81 | 0.723 | 0.60 | 8.83 |
| ridge | mean | 0.07 | 2423 | 81 | 0.723 | 0.60 | 10.59 |
| elasticnet | recency_weighted | 0.08 | 2423 | 81 | 0.722 | 0.58 | 9.61 |
| xgboost | val_weighted | 2.79 | 2423 | 81 | 0.717 | 0.62 | 0.26 |
| xgboost | mean | 1.65 | 2423 | 81 | 0.716 | 0.62 | 0.43 |
| lightgbm | mean | 0.79 | 2423 | 81 | 0.715 | 0.60 | 0.91 |
| lightgbm | val_weighted | 1.70 | 2423 | 81 | 0.714 | 0.60 | 0.42 |
| random_forest | val_weighted | 5.46 | 2423 | 81 | 0.713 | 0.60 | 0.13 |
| random_forest | mean | 2.80 | 2423 | 81 | 0.713 | 0.60 | 0.25 |
| random_forest | recency_weighted | 2.80 | 2423 | 81 | 0.707 | 0.58 | 0.25 |
| lightgbm | recency_weighted | 0.91 | 2423 | 81 | 0.705 | 0.60 | 0.78 |
| xgboost | recency_weighted | 1.47 | 2423 | 81 | 0.704 | 0.58 | 0.48 |
| ridge | recency_weighted | 0.07 | 2423 | 81 | 0.704 | 0.60 | 9.49 |

## Figures

![Spearman by window](figures/report_football_wr_windows.png)

![Model vs market](figures/report_football_wr_vs_market.png)
