# Position Predictor — Results: football RB v4

_**RB v4** · migrate-nflreadpy @ b320cd7-dirty_

_Generated 2026-06-23._ Test seasons **[2019, 2022, 2023, 2024, 2025]**, eligibility cutoff **g\* = 4 games**. Ranking is computed within each test season; metrics are mean ± sd across the season folds.

## Headline

- **Best model:** `random_forest` (mean, 20-yr window) — Spearman **0.819 ± 0.034**, Precision@12 0.50, MAE 2.61 PPG (full eligible universe).
- **Best baseline:** `linear` — Spearman 0.803 (the must-beat floor).
- **Market head-to-head** (FantasyPros preseason ECR, scored on the identical rows the market ranks): market Spearman 0.824 vs our best `xgboost` 0.793 — the model does **not** beat the market on overall rank.
  On **Precision@12 (tier-1 / RB1)**: market 0.62 vs best `elasticnet` 0.56 — model trails market.
  On **Precision@24 (tier-2 / RB2)**: market 0.75 vs best `elasticnet` 0.77 — model beats market.
  On the **top-weighted** rank score (Weighted τ — errors near #1 count most): market 0.743 vs `xgboost` 0.704 — the model does **not** beat the market where it matters most.

## Recency: how much history helps (§6.2)

Spearman by training-window length (years), at g\*:

| model | combine | 10yr | 20yr | 30yr |
|---|---|---|---|---|
| elasticnet | mean | 0.812 | 0.810 | 0.810 |
| elasticnet | recency_weighted | 0.811 | 0.808 | 0.808 |
| elasticnet | val_weighted | 0.812 | 0.810 | 0.810 |
| lasso | mean | 0.812 | 0.809 | 0.809 |
| lasso | recency_weighted | 0.811 | 0.810 | 0.810 |
| lasso | val_weighted | 0.812 | 0.810 | 0.810 |
| lightgbm | mean | 0.811 | 0.815 | 0.815 |
| lightgbm | recency_weighted | 0.806 | 0.809 | 0.809 |
| lightgbm | val_weighted | 0.811 | 0.815 | 0.815 |
| random_forest | mean | 0.814 | 0.819 | 0.819 |
| random_forest | recency_weighted | 0.812 | 0.815 | 0.815 |
| random_forest | val_weighted | 0.814 | 0.819 | 0.819 |
| ridge | mean | 0.808 | 0.807 | 0.807 |
| ridge | recency_weighted | 0.806 | 0.804 | 0.804 |
| ridge | val_weighted | 0.808 | 0.808 | 0.808 |
| xgboost | mean | 0.811 | 0.815 | 0.815 |
| xgboost | recency_weighted | 0.805 | 0.807 | 0.807 |
| xgboost | val_weighted | 0.811 | 0.815 | 0.815 |

> 20 yr ≈ 30 yr is expected — the box-score era is data-capped at 1999.

## Market benchmark (§7.4)

Preseason ECR coverage of the eligible universe and the market's own ranking quality, per test season:

| season | eligible | ranked | coverage | market Spearman | market P@12 |
|---|---|---|---|---|---|
| 2022 | 110 | 96 | 0.87 | 0.838 | 0.58 |
| 2023 | 104 | 88 | 0.85 | 0.832 | 0.42 |
| 2024 | 100 | 98 | 0.98 | 0.841 | 0.67 |
| 2025 | 100 | 78 | 0.78 | 0.787 | 0.83 |

**Head-to-head on the identical ranked rows** (mean across folds). `weighted_tau` is the **top-weighted** rank score — errors near #1 count most:

| model | Spearman | Weighted τ (top) | Precision@12 (RB1) | Precision@24 (RB2) |
|---|---|---|---|---|
| market_ecr _(market)_ | 0.824 | 0.743 | 0.62 | 0.75 |
| xgboost | 0.793 | 0.704 | 0.54 | 0.74 |
| lightgbm | 0.792 | 0.701 | 0.56 | 0.74 |
| random_forest | 0.793 | 0.694 | 0.54 | 0.74 |
| ridge | 0.779 | 0.685 | 0.56 | 0.75 |
| lasso | 0.779 | 0.681 | 0.56 | 0.75 |
| elasticnet | 0.780 | 0.678 | 0.56 | 0.77 |

## NGS-block ablation (§7.3)

`ngs` era model, longest window — **with** NGS: Spearman 0.777, P@12 0.45; **without**: Spearman 0.781, P@12 0.43.
- **Decision:** keep the `ngs_efficiency` block — it improves (coverage flags retained either way).

## Availability model (§7.4)

Predicting N+1 games played (gates projected eligibility / injury risk):

| model | games MAE | clears-cutoff AUC |
|---|---|---|
| baseline_prior_games | 4.52 | 0.784 |
| gbm_poisson | 5.13 | 0.736 |

## Eligibility-cutoff sensitivity (§6.4)

Best model (`random_forest`) Spearman across the candidate games cutoffs (robustness — the headline does not hinge on g\*):

| cutoff (games) | 4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|
| Spearman | 0.816 | 0.825 | 0.824 | 0.798 | 0.779 |


## Data volume

- **Feature rows:** 4464 (3064 labeled with a next-season target)
- **Features:** 81 columns
- **Seasons:** 1999–2025 (26 seasons)
- **Labeled rows per era:** boxscore 1714, snaps 501, ngs 849

## Compute & efficiency

- **Experiment wall-clock:** 85.6s (total model fit time 51.2s across the grid)

Per-model cost vs ranking quality at the headline window (30-yr, g\*=4). **Efficiency** = Spearman per fit-second:

| model | combine | fit (s) | train rows | features | Spearman | P@12 | efficiency |
|---|---|---|---|---|---|---|---|
| random_forest | mean | 2.71 | 2460 | 81 | 0.819 | 0.50 | 0.30 |
| random_forest | val_weighted | 5.20 | 2460 | 81 | 0.819 | 0.50 | 0.16 |
| random_forest | recency_weighted | 2.77 | 2460 | 81 | 0.815 | 0.48 | 0.29 |
| xgboost | val_weighted | 2.40 | 2460 | 81 | 0.815 | 0.50 | 0.34 |
| lightgbm | val_weighted | 1.22 | 2460 | 81 | 0.815 | 0.52 | 0.67 |
| xgboost | mean | 1.34 | 2460 | 81 | 0.815 | 0.50 | 0.61 |
| lightgbm | mean | 1.01 | 2460 | 81 | 0.815 | 0.50 | 0.81 |
| elasticnet | mean | 0.08 | 2460 | 81 | 0.810 | 0.52 | 10.41 |
| lasso | recency_weighted | 0.08 | 2460 | 81 | 0.810 | 0.50 | 10.73 |
| elasticnet | val_weighted | 0.16 | 2460 | 81 | 0.810 | 0.52 | 4.96 |
| lasso | val_weighted | 0.17 | 2460 | 81 | 0.810 | 0.52 | 4.78 |
| lightgbm | recency_weighted | 0.64 | 2460 | 81 | 0.809 | 0.47 | 1.27 |
| lasso | mean | 0.08 | 2460 | 81 | 0.809 | 0.52 | 9.86 |
| elasticnet | recency_weighted | 0.11 | 2460 | 81 | 0.808 | 0.50 | 7.61 |
| ridge | val_weighted | 0.13 | 2460 | 81 | 0.808 | 0.52 | 6.13 |
| ridge | mean | 0.06 | 2460 | 81 | 0.807 | 0.52 | 13.87 |
| xgboost | recency_weighted | 1.22 | 2460 | 81 | 0.807 | 0.48 | 0.66 |
| ridge | recency_weighted | 0.06 | 2460 | 81 | 0.804 | 0.50 | 13.26 |

## Figures

![Spearman by window](figures/report_football_rb_windows.png)

![Model vs market](figures/report_football_rb_vs_market.png)
