# Position Predictor — Results: football RB v4

_**RB v4** · primary @ 2bc2224-dirty_

_Generated 2026-07-06._ Test seasons **[2019, 2022, 2023, 2024, 2025]**, eligibility cutoff **g\* = 4 games**. Ranking is computed within each test season; metrics are mean ± sd across the season folds.

## Headline

- **Best model:** `random_forest` (val_weighted, 30-yr window) — Spearman **0.820 ± 0.035**, Precision@12 0.50, MAE 2.85 PPG (full eligible universe).
- **Best baseline:** `linear` — Spearman 0.800 (the must-beat floor).
- **Market head-to-head** (FantasyPros preseason ECR, scored on the identical rows the market ranks): market Spearman 0.822 vs our best `random_forest` 0.794 — the model does **not** beat the market on overall rank.
  On **Precision@12 (tier-1 / RB1)**: market 0.62 vs best `lightgbm` 0.58 — model trails market.
  On **Precision@24 (tier-2 / RB2)**: market 0.74 vs best `elasticnet` 0.75 — model beats market.
  On the **top-weighted** rank score (Weighted τ — errors near #1 count most): market 0.739 vs `xgboost` 0.691 — the model does **not** beat the market where it matters most.

## Recency: how much history helps (§6.2)

Spearman by training-window length (years), at g\*:

| model | combine | 10yr | 20yr | 30yr |
|---|---|---|---|---|
| elasticnet | mean | 0.811 | 0.810 | 0.810 |
| elasticnet | recency_weighted | 0.811 | 0.810 | 0.810 |
| elasticnet | val_weighted | 0.811 | 0.810 | 0.810 |
| lasso | mean | 0.811 | 0.809 | 0.809 |
| lasso | recency_weighted | 0.810 | 0.810 | 0.810 |
| lasso | val_weighted | 0.811 | 0.809 | 0.809 |
| lightgbm | mean | 0.810 | 0.816 | 0.816 |
| lightgbm | recency_weighted | 0.807 | 0.810 | 0.810 |
| lightgbm | val_weighted | 0.810 | 0.816 | 0.816 |
| random_forest | mean | 0.813 | 0.820 | 0.820 |
| random_forest | recency_weighted | 0.810 | 0.815 | 0.815 |
| random_forest | val_weighted | 0.812 | 0.820 | 0.820 |
| ridge | mean | 0.809 | 0.809 | 0.809 |
| ridge | recency_weighted | 0.806 | 0.805 | 0.805 |
| ridge | val_weighted | 0.810 | 0.809 | 0.809 |
| xgboost | mean | 0.808 | 0.811 | 0.811 |
| xgboost | recency_weighted | 0.806 | 0.806 | 0.806 |
| xgboost | val_weighted | 0.807 | 0.811 | 0.811 |

> 20 yr ≈ 30 yr is expected — the box-score era is data-capped at 1999.

## Market benchmark (§7.4)

Preseason ECR coverage of the eligible universe and the market's own ranking quality, per test season:

| season | eligible | ranked | coverage | market Spearman | market P@12 |
|---|---|---|---|---|---|
| 2022 | 110 | 96 | 0.87 | 0.831 | 0.58 |
| 2023 | 104 | 88 | 0.85 | 0.835 | 0.33 |
| 2024 | 100 | 98 | 0.98 | 0.838 | 0.67 |
| 2025 | 100 | 78 | 0.78 | 0.785 | 0.92 |

**Head-to-head on the identical ranked rows** (mean across folds). `weighted_tau` is the **top-weighted** rank score — errors near #1 count most:

| model | Spearman | Weighted τ (top) | Precision@12 (RB1) | Precision@24 (RB2) |
|---|---|---|---|---|
| market_ecr _(market)_ | 0.822 | 0.739 | 0.62 | 0.74 |
| xgboost | 0.786 | 0.691 | 0.56 | 0.72 |
| random_forest | 0.794 | 0.689 | 0.54 | 0.72 |
| lightgbm | 0.793 | 0.685 | 0.58 | 0.72 |
| ridge | 0.781 | 0.681 | 0.54 | 0.74 |
| lasso | 0.779 | 0.674 | 0.54 | 0.75 |
| elasticnet | 0.781 | 0.671 | 0.54 | 0.75 |

## NGS-block ablation (§7.3)

`ngs` era model, longest window — **with** NGS: Spearman 0.778, P@12 0.45; **without**: Spearman 0.777, P@12 0.45.
- **Decision:** keep the `ngs_efficiency` block — it improves (coverage flags retained either way).

## Availability model (§7.4)

Predicting N+1 games played (gates projected eligibility / injury risk):

| model | games MAE | clears-cutoff AUC |
|---|---|---|
| baseline_prior_games | 4.52 | 0.784 |
| gbm_poisson | 5.12 | 0.777 |

## Eligibility-cutoff sensitivity (§6.4)

Best model (`random_forest`) Spearman across the candidate games cutoffs (robustness — the headline does not hinge on g\*):

| cutoff (games) | 4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|
| Spearman | 0.816 | 0.824 | 0.823 | 0.796 | 0.779 |


## Data volume

- **Feature rows:** 4464 (3064 labeled with a next-season target)
- **Features:** 81 columns
- **Seasons:** 1999–2025 (26 seasons)
- **Labeled rows per era:** boxscore 1714, snaps 501, ngs 849

## Compute & efficiency

- **Experiment wall-clock:** 63.9s (total model fit time 46.0s across the grid)

Per-model cost vs ranking quality at the headline window (30-yr, g\*=4). **Efficiency** = Spearman per fit-second:

| model | combine | fit (s) | train rows | features | Spearman | P@12 | efficiency |
|---|---|---|---|---|---|---|---|
| random_forest | val_weighted | 3.28 | 2460 | 81 | 0.820 | 0.50 | 0.25 |
| random_forest | mean | 1.44 | 2460 | 81 | 0.820 | 0.50 | 0.57 |
| lightgbm | mean | 1.54 | 2460 | 81 | 0.816 | 0.53 | 0.53 |
| lightgbm | val_weighted | 2.91 | 2460 | 81 | 0.816 | 0.53 | 0.28 |
| random_forest | recency_weighted | 1.37 | 2460 | 81 | 0.815 | 0.48 | 0.59 |
| xgboost | val_weighted | 1.52 | 2460 | 81 | 0.811 | 0.52 | 0.53 |
| xgboost | mean | 0.79 | 2460 | 81 | 0.811 | 0.52 | 1.02 |
| elasticnet | mean | 0.05 | 2460 | 81 | 0.810 | 0.50 | 17.36 |
| elasticnet | recency_weighted | 0.05 | 2460 | 81 | 0.810 | 0.52 | 15.42 |
| elasticnet | val_weighted | 0.11 | 2460 | 81 | 0.810 | 0.50 | 7.72 |
| lightgbm | recency_weighted | 1.53 | 2460 | 81 | 0.810 | 0.52 | 0.53 |
| lasso | recency_weighted | 0.05 | 2460 | 81 | 0.810 | 0.52 | 15.62 |
| lasso | val_weighted | 0.10 | 2460 | 81 | 0.809 | 0.50 | 7.80 |
| ridge | val_weighted | 0.05 | 2460 | 81 | 0.809 | 0.50 | 15.78 |
| ridge | mean | 0.02 | 2460 | 81 | 0.809 | 0.50 | 34.71 |
| lasso | mean | 0.05 | 2460 | 81 | 0.809 | 0.50 | 17.54 |
| xgboost | recency_weighted | 0.79 | 2460 | 81 | 0.806 | 0.48 | 1.02 |
| ridge | recency_weighted | 0.02 | 2460 | 81 | 0.805 | 0.50 | 34.83 |

## Figures

![Spearman by window](figures/report_football_rb_windows.png)

![Model vs market](figures/report_football_rb_vs_market.png)
